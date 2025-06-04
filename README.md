# Full-Stack Machine Learning

## Intro

Dieses Repository enthält das Begleitmaterial für den Full Stack Machine Learning Kurs (Digethic Data Scientist / AI-Engineer).

Alle Notebooks unter `/notepads` sind strukturiert und können über die Ordnernummer und den Notebook-Code identifiziert werden. Alle Notebooks entsprechen den Folien und Videos, die für diesen Kurs erstellt wurden.

![image](https://user-images.githubusercontent.com/29402504/137859990-054ce9a4-f2d2-4054-8d25-faae4a466c5f.png)

Z.B. dieser Bezeichner verweist auf Ordner 2 und das Notebook mit dem Code EDA.

## Setup

### Linux- und Mac-Benutzer

- Führen Sie das Setup-Skript aus: `./setup.sh` oder `sh setup.sh`

### Windows-Benutzer

- Führen Sie das Setup-Skript aus: `.\setup.ps1`
- Wenn das Ausführen des Skripts aufgrund von Zugriffsrechten nicht funktioniert, versuchen Sie folgenden Befehl in Ihrem Terminal: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

## Entwicklung (Development)

- Mac/Linux: Aktivieren der Python-Umgebung: `source .venv/bin/activate`
- Windows: Aktivieren der Python-Umgebung: `.\.venv\Scripts\Activate.ps1`
- Python-Skript ausführen: `python <dateiname.py>`, z.B. `python train.py`
- Neue Abhängigkeit installieren: `pip install sklearn`
- Aktuell installierte Abhängigkeiten zurück in requirements.txt speichern: `pip freeze > requirements.txt`
- Um Jupyter Lab zu starten, führen Sie aus: `jupyter lab --ip=127.0.0.1 --port=8888`


# Zusätzliche Kursressourcen 🌍

Diese Liste enthält ergänzende Materialien wie Bücher, Artikel, Online-Kurse und Videos, um Ihr Verständnis für Data Science- und KI-Themen zu vertiefen.

---

## Allgemeine Übersichten zu Data Science & Machine Learning 📚

* **Bücher**:
    * "Python for Data Analysis" von Wes McKinney: Ein praktischer Leitfaden zur Datenmanipulation mit Pandas. Unverzichtbar für Python-basierte Data Science. (Deutscher Titel oft: "Python zur Datenanalyse")
    * "Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow" von Aurélien Géron: Ein umfassendes und sehr beliebtes Buch, das eine breite Palette von ML-Konzepten und Implementierungen abdeckt. (Deutscher Titel oft: "Praxiseinstieg Machine Learning mit Scikit-Learn, Keras & TensorFlow")
    * "An Introduction to Statistical Learning (with Applications in R or Python)" von Gareth James, Daniela Witten, Trevor Hastie und Robert Tibshirani: Bietet einen klaren und zugänglichen Überblick über statistische Lernmethoden. Die Python-Version ist besonders relevant. (Kostenloses PDF online verfügbar)
    * "The Elements of Statistical Learning" von Trevor Hastie, Robert Tibshirani und Jerome Friedman: Ein fortgeschritteneres und umfassenderes Werk, oft als eine Bibel des maschinellen Lernens betrachtet. (Kostenloses PDF online verfügbar)
    * "Pattern Recognition and Machine Learning" von Christopher M. Bishop: Ein klassisches, eher theoretisches Lehrbuch zum maschinellen Lernen.
* **Websites & Blogs**:

    * [Towards Data Science](https://towardsdatascience.com/): Eine Medium-Publikation mit einer riesigen Sammlung von Artikeln zu Data Science, ML und KI.
    * [KDnuggets](https://www.kdnuggets.com/): Eine führende Seite zu KI, Analytik, Big Data, Data Mining, Data Science und Machine Learning.
    * [Distill.pub](https://distill.pub/): Veröffentlicht klare, interaktive Artikel, die Konzepte des maschinellen Lernens erklären.
    * [Google AI Blog](https://ai.googleblog.com/): Updates und Einblicke aus der KI-Forschung von Google.
    * [OpenAI Blog](https://openai.com/blog/): Forschung und Ankündigungen von OpenAI.
    * [Machine Learning Mastery](https://machinelearningmastery.com/): Artikel und Tutorials zu allen möglichen Machine Learning Themen.

---

## Python für Data Science 🐍

* **Bücher**:
    * "Python for Data Analysis" von Wes McKinney (bereits erwähnt, aber hier entscheidend). (Deutscher Titel oft: "Python zur Datenanalyse")
    * "Fluent Python" von Luciano Ramalho: Für diejenigen, die idiomatischeren und effizienteren Python-Code schreiben möchten.
* **Tutorials & Dokumentation**:
    * Offizielles Python-Tutorial: [docs.python.org/3/tutorial/](https://docs.python.org/3/tutorial/)
    * NumPy-Dokumentation: [numpy.org/doc/stable/](https://numpy.org/doc/stable/)
    * Pandas-Dokumentation: [pandas.pydata.org/pandas-docs/stable/](https://pandas.pydata.org/pandas-docs/stable/)
    * Scikit-learn-Dokumentation: [scikit-learn.org/stable/documentation.html](https://scikit-learn.org/stable/documentation.html)
* **Videos**:
    * Corey Schafer's Python YouTube-Kanal: Ausgezeichnete Tutorials zu verschiedenen Python-Themen, einschließlich Pandas und OOP.
    * Sentdex YouTube-Kanal: Behandelt eine breite Palette von Python-Programmierung, einschließlich Datenanalyse und maschinellem Lernen.

---

## Statistische Grundlagen & EDA (Explorative Datenanalyse) 📊

* **Bücher**:
    * "Think Stats: Exploratory Data Analysis in Python" von Allen B. Downey (Kostenloses PDF online verfügbar).
    * "Practical Statistics for Data Scientists" von Peter Bruce, Andrew Bruce & Peter Gedeck. (Deutscher Titel oft: "Statistik für Data Scientists: Praxiswissen für den Berufsalltag")
* **Online-Kurse**:
    * Khan Academy's Statistics and Probability: [khanacademy.org/math/statistics-probability](https://khanacademy.org/math/statistics-probability) (Viele Inhalte auch auf Deutsch verfügbar)
* **Artikel**:
    * "Understanding Descriptive and Inferential Statistics" - Viele gute Artikel finden sich auf Towards Data Science oder ähnlichen Blogs.
Deutsch verfügbar)
* **Videos**:
    * "3Blue1Brown Youtube-Kanal": Statistische und mathematische Grundlagen gut erklärt und visualisiert.

---

## Datenvorverarbeitung (Data Preprocessing) 🛠️

* **Artikel & Dokumentation**:
    * Scikit-learn-Dokumentation zu [Preprocessing data](https://scikit-learn.org/stable/modules/preprocessing.html).
    * Artikel über Feature Scaling, Kodierung kategorialer Daten, Umgang mit fehlenden Werten (z.B. auf KDnuggets, Towards Data Science).
* **Videos**:
    * StatQuest with Josh Starmer (YouTube): Hat ausgezeichnete, leicht verständliche Videos zu verschiedenen statistischen und ML-Konzepten, einschließlich derer, die für die Vorverarbeitung relevant sind, wie PCA (was Multikollinearität berührt).

---

## Kernalgorithmen des Maschinellen Lernens 🧠

### k-Nächste-Nachbarn (kNN)
* **Videos**:
    * StatQuest: k-nearest neighbors (kNN) clearly explained (Suche nach dem neuesten Video)
* **Artikel**:
    * "A Detailed Introduction to K-Nearest Neighbor (KNN) Algorithm" auf Towards Data Science.

### Lineare & Polynomiale Regression
* **Bücher**:
    * Kapitel in "An Introduction to Statistical Learning."
* **Videos**:
    * StatQuest: Linear Regression, Clearly Explained!!!
    * Khan Academy Videos zur linearen Regression.

### Logistische Regression
* **Videos**:
    * StatQuest: Logistic Regression
* **Artikel**:
    * "Logistic Regression: Detailed Overview" auf Towards Data Science.

### Entscheidungsbäume & Ensemble-Methoden (Random Forest, AdaBoost)
* **Bücher**:
    * Kapitel in "Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow." (Deutscher Titel oft: "Praxiseinstieg Machine Learning mit Scikit-Learn, Keras & TensorFlow")
* **Videos**:
    * StatQuest: Decision Trees, Random Forests, AdaBoost, Gradient Boost (XGBoost) - alle haben eigene Videos.
* **Artikel**:
    * "Understanding Random Forest" auf Towards Data Science.
    * "A Gentle Introduction to AdaBoost" auf Machine Learning Mastery.

### Support Vector Machines (SVM)
* **Videos**:
    * StatQuest: Support Vector Machines (SVMs), Clearly Explained
* **Artikel**:
    * "Understanding Support Vector Machine (SVM) algorithm" auf Towards Data Science.

### Naive Bayes
* **Videos**:
    * StatQuest: Naive Bayes, Clearly Explained
* **Artikel**:
    * "Naive Bayes Classifier Explained" auf Towards Data Science.

---

## Modellbewertung & -verbesserung ⚙️✨

* **Dokumentation & Artikel**:
    * Scikit-learn-Dokumentation zu [Model evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html) (ROC, AUC, etc.).
    * Scikit-learn-Dokumentation zu [Tuning the hyper-parameters of an estimator](https://scikit-learn.org/stable/modules/grid_search.html).
    * "Understanding AUC - ROC Curve" auf Towards Data Science.
    * "Hyperparameter Tuning Techniques in Machine Learning" auf diversen Blogs.
    * "A Short Introduction to the Curse of Dimensionality" von Lilian Weng.
* **Videos**:
    * StatQuest: ROC and AUC, Clearly Explained!
    * Videos zu Kreuzvalidierung und Hyperparameter-Tuning (z.B. von Sentdex oder Krish Naik auf YouTube).

### Modellinterpretation (SHAP)
* **Bücher/Paper**:
    * Das ursprüngliche SHAP-Paper: "A Unified Approach to Interpreting Model Predictions" von Lundberg und Lee.
* **Dokumentation & Artikel**:
    * SHAP GitHub Repository und Dokumentation: [github.com/slundberg/shap](https://github.com/slundberg/shap)
    * "Explain Your Model with SHAP Values" auf Towards Data Science.

---

## Deep Learning 💡

* **Bücher**:
    * "Deep Learning" von Ian Goodfellow, Yoshua Bengio und Aaron Courville (Das "Deep Learning Buch" - kostenlos online).
    * "Deep Learning with Python" von François Chollet (Entwickler von Keras). (Deutscher Titel oft: "Deep Learning mit Python und Keras")
* **Online-Kurse**:
    * fast.ai Kurse (oben erwähnt).
    * Udacity's Deep Learning Nanodegree.
* **Videos**:
    * 3Blue1Brown YouTube-Kanal: Serie zu Neuronalen Netzen für ein intuitives Verständnis.
    * Lex Fridman Podcast: Interviews mit führenden KI-Forschern (eher für breiteren Kontext und Inspiration).
    * Two Minute Papers YouTube-Kanal: Fasst spannende neue KI-Forschungsarbeiten zusammen.

### PyTorch
* **Dokumentation & Tutorials**:
    * Offizielle PyTorch-Tutorials: [pytorch.org/tutorials/](https://pytorch.org/tutorials/)
* **Online-Kurse**:
    * Udacity: "Intro to Deep Learning with PyTorch".
    * DeepLearning.AI PyTorch Kurse auf Coursera.

### Convolutional Neural Networks (CNN)
* **Artikel**:
    * "A Comprehensive Guide to Convolutional Neural Networks" auf Towards Data Science.
    * Stanford CS231n: Convolutional Neural Networks for Visual Recognition (Kursnotizen sind ausgezeichnet und frei online verfügbar).
* **Videos**:
    * Viele Videos auf YouTube, die CNN-Architekturen erklären (z.B. LeNet, AlexNet, VGG, ResNet).

### Representation Learning (Autoencoder, PCA, t-SNE)
* **Artikel**:
    * "Understanding Autoencoders" auf Towards Data Science.
    * "Principal Component Analysis (PCA) Explained" auf diversen Blogs. (Hauptkomponentenanalyse auf Deutsch)
    * "How to Use t-SNE Effectively" auf Distill.pub.
* **Videos**:
    * StatQuest: Principal Component Analysis (PCA), Step-by-Step
    * StatQuest: t-SNE, Clearly Explained

---

## Quellen für Datensätze 💾

Zusätzlich zu den im Kurs bereitgestellten Datensätzen gibt es viele öffentliche Quellen für Datensätze, die für Projekte und zum Üben verwendet werden können:

* **Huggingface**: [huggingface.com](https://huggingface.co/) - Datensätze und Modelle für Machine Learning
* **Kaggle Datasets**: [kaggle.com/datasets](https://www.kaggle.com/datasets) - Eine sehr große Sammlung von Datensätzen zu verschiedensten Themen.
* **UCI Machine Learning Repository**: [archive.ics.uci.edu/ml/index.php](https://archive.ics.uci.edu/ml/index.php) - Ein klassisches Repository für ML-Datensätze.
* **Google Dataset Search**: [datasetsearch.research.google.com](https://datasetsearch.research.google.com) - Eine Suchmaschine für Datensätze.
* **Data.gov**: [www.data.gov](https://www.data.gov) - Datensätze der US-Regierung.
* **EU Open Data Portal**: [data.europa.eu/euodp/de/home](https://data.europa.eu/euodp/de/home) - Datensätze von EU-Institutionen.
* **Statistisches Bundesamt (Destatis)**: [www.destatis.de](https://www.destatis.de) - Für deutsche Statistiken.
* **Awesome Public Datasets (GitHub)**: [github.com/awesomedata/awesome-public-datasets](https://github.com/awesomedata/awesome-public-datasets) - Eine kuratierte Liste öffentlicher Datensätze.

---

## Newsletter 📰

Bleiben Sie auf dem Laufenden mit diesen Newslettern:

* **Data Elixir**: [dataelixir.com](https://dataelixir.com/) - Kuratierte Nachrichten und Ressourcen zu Data Science.
* **KDnuggets News**: [kdnuggets.com/news/subscribe.html](https://www.kdnuggets.com/news/subscribe.html) - Wöchentliche Zusammenfassung von KDnuggets.
* **The Batch (DeepLearning.AI)**: [deeplearning.ai/the-batch/](https://www.deeplearning.ai/the-batch/) - Wöchentliche KI-Nachrichten von Andrew Ng's Team.
* **Import AI**: [jack-clark.net](https://jack-clark.net/) - Wöchentlicher Newsletter über KI-Forschung und -Entwicklungen.
* **O'Reilly Data & AI Newsletter**: [oreilly.com/content-marketing/newsletter/](https://www.oreilly.com/content-marketing/newsletter/) (Suche nach Data & AI)

---

## Diskussionsforen und Communitys 🗣️💬

Tauschen Sie sich mit anderen aus und stellen Sie Fragen:

* **Stack Overflow**: [stackoverflow.com](https://stackoverflow.com) (Tags: `python`, `pandas`, `scikit-learn`, `tensorflow`, `pytorch`, `machine-learning`, `deep-learning`)
* **Cross Validated (Stack Exchange)**: [stats.stackexchange.com](https://stats.stackexchange.com) - Für Fragen zu Statistik und maschinellem Lernen.
* **Kaggle Discussions**: [kaggle.com/discussions](https://www.kaggle.com/discussions) - Diskussionsforen zu Wettbewerben, Datensätzen und allgemeinen ML-Themen.
* **Reddit**:
    * r/MachineLearning: [reddit.com/r/MachineLearning/](https://www.reddit.com/r/MachineLearning/)
    * r/datascience: [reddit.com/r/datascience/](https://www.reddit.com/r/datascience/)
    * r/learnmachinelearning: [reddit.com/r/learnmachinelearning/](https://www.reddit.com/r/learnmachinelearning/)

---

## Möglichkeiten für freiwilliges Engagement & Open Source Projekte 🤝💡

Tragen Sie zu Projekten bei und sammeln Sie praktische Erfahrung:

* **GitHub**: [github.com](https://github.com) - Suchen Sie nach Projekten mit Tags wie `good first issue`, `help wanted` in Bereichen wie `scikit-learn`, `pandas`, `tensorflow`, `pytorch` oder anderen Bibliotheken, die Sie interessieren.
* **Kaggle Wettbewerbe**: [kaggle.com/competitions](https://www.kaggle.com/competitions) - Nehmen Sie an Wettbewerben teil, um Ihre Fähigkeiten zu testen und von anderen zu lernen. Oft gibt es auch Team-Möglichkeiten.
* **DataKind**: [datakind.org](https://www.datakind.org) - Organisationen, die Data Science im Dienste der Menschheit einsetzen (Projekte oft für erfahrene Fachleute, aber es gibt auch Möglichkeiten, sich zu engagieren).
* **Omdena**: [omdena.com](https://omdena.com/) - Kollaborative KI-Projekte zur Lösung realer Probleme.
* **Zooniverse**: [zooniverse.org](https://www.zooniverse.org) - Bürgerwissenschaftliche Projekte, einige davon beinhalten Datenanalyse oder Bilderkennung.

---

## Konferenzen (Auswahl) 📅🏛️

Konferenzen sind eine großartige Möglichkeit, sich über die neuesten Entwicklungen zu informieren und Kontakte zu knüpfen (viele bieten auch Online-Zugang oder Aufzeichnungen):

* **Europa / Deutschland (Beispiele)**:
    * **PyData Conferences**: [pydata.org](https://pydata.org/) (weltweit, auch in Europa, z.B. Berlin, Amsterdam)
    * **ODSC (Open Data Science Conference) Europe**: [odsc.com/europe/](https://odsc.com/europe/)
    * **Data Natives**: [datanatives.io](https://datanatives.io/) (oft in Berlin)
    * Achten Sie auf lokale Universitäts-Workshops und Industrieveranstaltungen.

---
