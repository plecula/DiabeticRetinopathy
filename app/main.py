import os
import datetime
from flask import Flask, render_template, request, redirect, url_for, flash, send_file
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin, LoginManager, login_required, login_user, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
# ----------------- KONFIGURACJA APLIKACJI -----------------

app = Flask(__name__)

# klucz do sesji / flash – zmień na coś własnego
with open("key.txt") as f:
    app.config["SECRET_KEY"] = f.read().strip()


# baza SQLite w pliku database.db obok main.py
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///database.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

# folder na przesłane obrazy
UPLOAD_FOLDER = os.path.join("static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ----------------- MODELE BAZY -----------------

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(80))  # imię / nazwa użytkownika
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)

    analyses = db.relationship("AnalysisRecord", backref="user", lazy=True)

    def __repr__(self):
        return f"<User {self.email}>"




class AnalysisRecord(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    result = db.Column(db.Integer)  # 0 = brak cech, 1 = podejrzenie
    score = db.Column(db.Float)  # np. prawdopodobieństwo modelu
    image_path = db.Column(db.String(255))
    pdf_url = db.Column(db.String(255))

    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)

    def __repr__(self):
        return f"<AnalysisRecord {self.id} user={self.user_id}>"


# ----------------- FLASK-LOGIN -----------------

login_manager = LoginManager(app)
login_manager.login_view = "login"  # niezalogowany trafi na /login


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


# ----------------- MIEJSCE NA MODEL -----------------
# Tu możesz załadować swój model TensorFlow i napisać funkcję,
# która zwróci (prediction, score) dla podanej ścieżki obrazu.

# ----------------- Pytorch model -----------------

# ----------------- PYTORCH MODEL: ConvNeXt-Tiny -----------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = os.path.join("models", "best_convnext_adamw_ce_epochs30.pth")
IMG_SIZE = (224, 224)  # jak w val_transform


def build_convnext_tiny_infer():
    # architektura TAKA SAMA jak w trenowaniu
    model = models.convnext_tiny(weights=None)  # pretrained niepotrzebne, bo i tak wczytujemy własne wagi
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, 2)
    return model


# tworzymy model i ładujemy wagi ze .pth
model = build_convnext_tiny_infer()
state = torch.load(MODEL_PATH, map_location=DEVICE)
# w trenowaniu było: torch.save(model.state_dict(), best_model_path)
# więc tu ładujemy bez klucza 'model_state_dict'
model.load_state_dict(state)
model.to(DEVICE)
model.eval()

# preprocess jak val_transform
transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
])

def run_model_on_image(image_path: str):
    # 1. Wczytanie obrazu
    img = Image.open(image_path).convert("RGB")
    x = transform(img)             # [C, H, W]
    x = x.unsqueeze(0).to(DEVICE)  # [1, C, H, W]

    # 2. Przepuszczenie przez model
    with torch.no_grad():
        outputs = model(x)         # shape [1, 2] – logity

    # 3. Softmax → prawdopodobieństwa 2 klas
    probs = torch.softmax(outputs, dim=1)[0]
    prob_disease = probs[1].item()   # klasa 1 = "choroba", dokładnie jak w trenowaniu

    # 4. Próg decyzyjny
    threshold = 0.5
    prediction = 1 if prob_disease >= threshold else 0

    return prediction, float(prob_disease)

# ----------------- ROUTES -----------------


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["GET", "POST"])
@login_required
def upload():
    image_url = None
    result_text = None
    error = None

    if request.method == "POST":
        file = request.files.get("file")

        if not file or file.filename == "":
            error = "Nie wybrano pliku."
            return render_template(
                "upload.html", image_url=None, result=None, error=error
            )

        # zapis pliku
        filename = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_") + file.filename
        save_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(save_path)

        # ścieżka do wyświetlenia w HTML
        image_url = url_for("static", filename=f"uploads/{filename}")

        # wywołanie modelu
        prediction, score = run_model_on_image(save_path)

        # tekst wyniku
        if prediction == 1:
            result_text = f"PODEJRZENIE RETINOPATII (p={score:.2f})"
        else:
            result_text = f"BRAK CECH RETINOPATII (p={score:.2f})"

        # zapis rekordu w bazie
        record = AnalysisRecord(
            result=prediction,
            score=score,
            image_path=image_url,  # zapisujemy URL do późniejszego użycia
            pdf_url=None,
            user_id=current_user.id,
        )
        db.session.add(record)
        db.session.commit()

        return render_template(
            "upload.html", image_url=image_url, result=result_text, error=None
        )

    return render_template(
        "upload.html", image_url=image_url, result=result_text, error=error
    )


@app.route("/account")
@login_required
def account():
    records = (
        AnalysisRecord.query.filter_by(user_id=current_user.id)
        .order_by(AnalysisRecord.date.desc())
        .all()
    )
    return render_template("account.html", records=records)

@app.route("/account/pdf")
@login_required
def download_all_pdf():
    records = (
        AnalysisRecord.query
        .filter_by(user_id=current_user.id)
        .order_by(AnalysisRecord.date.asc())
        .all()
    )

    if not records:
        # nie ma co generować
        abort(404)

    buffer = BytesIO()
    p = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    # nagłówek
    p.setFont("Helvetica-Bold", 16)
    p.drawString(72, height - 72, "Zbiorczy raport analiz dna oka")

    p.setFont("Helvetica", 11)
    p.drawString(72, height - 100, f"Użytkownik: {current_user.name or current_user.email}")
    p.drawString(72, height - 116, f"Liczba badań: {len(records)}")

    y = height - 150
    p.setFont("Helvetica-Bold", 11)
    p.drawString(72, y, "Data badania")
    p.drawString(220, y, "Wynik")
    p.drawString(400, y, "Ocena modelu")
    p.setFont("Helvetica", 11)
    y -= 18

    for r in records:
        if y < 72:  # nowa strona
            p.showPage()
            y = height - 72
            p.setFont("Helvetica-Bold", 11)
            p.drawString(72, y, "Data badania")
            p.drawString(220, y, "Wynik")
            p.drawString(400, y, "Ocena modelu")
            p.setFont("Helvetica", 11)
            y -= 18

        date_str = r.date.strftime("%Y-%m-%d %H:%M") if r.date else "-"
        status = "Podejrzenie retinopatii" if r.result == 1 else "Brak cech retinopatii"
        score_str = f"{r.score:.2f}" if r.score is not None else "-"

        p.drawString(72, y, date_str)
        p.drawString(220, y, status)
        p.drawString(400, y, score_str)
        y -= 16

    p.showPage()
    p.save()
    buffer.seek(0)

    return send_file(
        buffer,
        as_attachment=True,
        download_name="raport_analiz.pdf",
        mimetype="application/pdf",
    )


@app.route("/record/<int:record_id>/pdf")
@login_required
def download_pdf(record_id):
    # szukamy rekordu tylko bieżącego użytkownika
    record = AnalysisRecord.query.filter_by(
        id=record_id,
        user_id=current_user.id
    ).first()

    if record is None:
        abort(404)

    buffer = BytesIO()
    p = canvas.Canvas(buffer, pagesizes=A4)
    width, height = A4

    p.setFont("Helvetica-Bold", 16)
    p.drawString(72, height - 72, "Raport analizy dna oka")

    p.setFont("Helvetica", 11)
    p.drawString(72, height - 110, f"Data badania: {record.date.strftime('%Y-%m-%d %H:%M')}")
    status = "Podejrzenie retinopatii" if record.result == 1 else "Brak cech retinopatii"
    p.drawString(72, height - 130, f"Wynik: {status}")
    if record.score is not None:
        p.drawString(72, height - 150, f"Ocena modelu (p): {record.score:.2f}")

    p.showPage()
    p.save()
    buffer.seek(0)

    return send_file(
        buffer,
        as_attachment=True,
        download_name=f"raport_{record.id}.pdf",
        mimetype="application/pdf"
    )


# ----------------- AUTORYZACJA -----------------


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form.get("name")
        email = request.form.get("email")
        password = request.form.get("password")
        password2 = request.form.get("password2")

        if not name or not email or not password or not password2:
            flash("Uzupełnij wszystkie pola.", "error")
            return redirect(url_for("register"))

        if password != password2:
            flash("Hasła nie są takie same.", "error")
            return redirect(url_for("register"))

        existing = User.query.filter_by(email=email).first()
        if existing:
            flash("Konto z takim adresem e-mail już istnieje.", "error")
            return redirect(url_for("register"))

        user = User(
            name=name,
            email=email,
            password_hash=generate_password_hash(password),
        )
        db.session.add(user)
        db.session.commit()

        flash("Konto zostało utworzone. Możesz się zalogować.", "success")
        return redirect(url_for("login"))

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")

        user = User.query.filter_by(email=email).first()
        if not user or not check_password_hash(user.password_hash, password):
            flash("Nieprawidłowy e-mail lub hasło.", "error")
            return redirect(url_for("login"))

        login_user(user)
        return redirect(url_for("account"))

    return render_template("login.html")


@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("index"))


# ----------------- START APLIKACJI -----------------

if __name__ == "__main__":
    with app.app_context():
        db.create_all()

    app.run(debug=True)
