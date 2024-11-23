from flask import Flask, render_template, request
import random
from langdetect import detect, LangDetectException

app = Flask(__name__)

class LanguageDetector:
    def __init__(self, text=""):
        self.text = text

    def set_text(self, text):
        self.text = text

    def detect_language(self):
        try:
            language_code = detect(self.text)
            return language_code
        except LangDetectException:
            return "English"

# Define translations for each language
translations = {
    'English': {
        'title': 'Submit Book Details',
        'book_name': 'Name of the Book',
        'language': 'Language',
        'category': 'Category',
        'synopsis': 'Synopsis (Optional)',
        'submit': 'Submit',
        'feedback_title': 'Book Submission Result',
        'name_label': 'Book Name:',
        'language_label': 'Language:',
        'category_label': 'Category:',
        'synopsis_label': 'Synopsis:',
        'publisher_label': 'Publisher:',
        'probability_label': 'Probability of Accepting:',
        'related_book_label': 'Related Book Published:',
        'submit_button': 'Submit another book'
    },
    'Dutch': {
        'title': 'Boekdetails indienen',
        'book_name': 'Naam van het boek',
        'language': 'Taal',
        'category': 'Categorie',
        'synopsis': 'Samenvatting (Optioneel)',
        'submit': 'Verzenden',
        'feedback_title': 'Resultaat van boekinzending',
        'name_label': 'Boeknaam:',
        'language_label': 'Taal:',
        'category_label': 'Categorie:',
        'synopsis_label': 'Samenvatting:',
        'publisher_label': 'Uitgever:',
        'probability_label': 'Aanvaardingskans:',
        'related_book_label': 'Gerelateerd boek gepubliceerd:',
        'submit_button': 'Nog een boek indienen'
    },
    'French': {
        'title': 'Soumettre les détails du livre',
        'book_name': 'Nom du livre',
        'language': 'Langue',
        'category': 'Catégorie',
        'synopsis': 'Résumé (Facultatif)',
        'submit': 'Soumettre',
        'feedback_title': 'Résultat de la soumission du livre',
        'name_label': 'Nom du livre:',
        'language_label': 'Langue:',
        'category_label': 'Catégorie:',
        'synopsis_label': 'Résumé:',
        'publisher_label': 'Éditeur:',
        'probability_label': 'Probabilité d\'acceptation:',
        'related_book_label': 'Livre publié similaire:',
        'submit_button': 'Soumettre un autre livre'
    },
    'Haitian Creole': {
        'title': 'Soumèt detay liv la',
        'book_name': 'Non liv la',
        'language': 'Lang',
        'category': 'Kategori',
        'synopsis': 'Rezime (Opsyonèl)',
        'submit': 'Soumèt',
        'feedback_title': 'Rezilta soumèt liv la',
        'name_label': 'Non liv la:',
        'language_label': 'Lang:',
        'category_label': 'Kategori:',
        'synopsis_label': 'Rezime:',
        'publisher_label': 'Piblikatè:',
        'probability_label': 'Pwobabilite Akseptasyon:',
        'related_book_label': 'Liv ki gen rapò pibliye:',
        'submit_button': 'Soumèt yon lòt liv'
    },
    'Papiamentu': {
        'title': 'Entregá detaye di e buki',
        'book_name': 'Nòm di e buki',
        'language': 'Idioma',
        'category': 'Kategoria',
        'synopsis': 'Resumenchi (Opshonal)',
        'submit': 'Entregá',
        'feedback_title': 'Resultado di entrega di e buki',
        'name_label': 'Nòm di e buki:',
        'language_label': 'Idioma:',
        'category_label': 'Kategoria:',
        'synopsis_label': 'Resumenchi:',
        'publisher_label': 'Editoria:',
        'probability_label': 'Probabilidat di akseptashon:',
        'related_book_label': 'Buki relashona publika:',
        'submit_button': 'Entregá otro buki'
    },
    'Spanish': {
        'title': 'Enviar detalles del libro',
        'book_name': 'Nombre del libro',
        'language': 'Idioma',
        'category': 'Categoría',
        'synopsis': 'Sinopsis (Opcional)',
        'submit': 'Enviar',
        'feedback_title': 'Resultado de la presentación del libro',
        'name_label': 'Nombre del libro:',
        'language_label': 'Idioma:',
        'category_label': 'Categoría:',
        'synopsis_label': 'Sinopsis:',
        'publisher_label': 'Editorial:',
        'probability_label': 'Probabilidad de aceptación:',
        'related_book_label': 'Libro relacionado publicado:',
        'submit_button': 'Enviar otro libro'
    }
}

@app.route('/', methods=['GET', 'POST'])
def home():
    selected_language = request.args.get('language', 'English')
    language_content = translations.get(selected_language, translations['English'])

    if request.method == 'POST':
        # Get form data
        book_name = request.form['book_name']
        language = request.form.get('language', '')  # Empty if not selected
        category = request.form['category']
        synopsis = request.form.get('synopsis', '')

        # Detect language if not selected
        if not language:
            # Use book_name first for detection; fall back to synopsis if book_name is empty
            detector = LanguageDetector(book_name or synopsis)
            language = detector.detect_language()

        # Prepare the data to display in the feedback page
        book_data = {
            'book_name': book_name,
            'language': language,
            'category': category,
            'synopsis': synopsis,
        }
        return render_template('result.html', book_data=book_data, translation=language_content)

    # Language and Category options
    languages = list(translations.keys())
    categories = ['Fiction', 'Non-Fiction', 'Science Fiction', 'Fantasy', 'Mystery', 'Biography']

    return render_template('index.html', languages=languages, categories=categories, selected_language=selected_language, language_content=language_content)


if __name__ == '__main__':
    app.run(debug=True)