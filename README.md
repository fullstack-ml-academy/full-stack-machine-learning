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
    * "An Introduction to Statistical Learning (with Applications in R or Python)" von Gareth James, Daniela Witten, Trevor Hastie und Robert Tibshirani: Bietet einen klaren und zugänglichen Überblick über statistische Lernmethoden. Die Python-Version ist besonders relevant. (Kostenloses PDF online verfügbar unter [www.statlearning.com](https://www.statlearning.com/))
    * "The Elements of Statistical Learning" von Trevor Hastie, Robert Tibshirani und Jerome Friedman: Ein fortgeschritteneres und umfassenderes Werk, oft als eine Bibel des maschinellen Lernens betrachtet. (Kostenloses PDF online verfügbar unter [hastie.su.domains/ElemStatLearn/](https://hastie.su.domains/ElemStatLearn/))
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
    * "Python for Data Analysis" von Wes McKinney 
    * "Fluent Python" von Luciano Ramalho: Für diejenigen, die idiomatischeren und effizienteren Python-Code schreiben möchten.
* **Tutorials & Dokumentation**:
    * Offizielles Python-Tutorial: [docs.python.org/3/tutorial/](https://docs.python.org/3/tutorial/)
    * NumPy-Dokumentation: [numpy.org/doc/stable/](https://numpy.org/doc/stable/)
    * Pandas-Dokumentation: [pandas.pydata.org/pandas-docs/stable/](https://pandas.pydata.org/pandas-docs/stable/)
    * Scikit-learn-Dokumentation: [scikit-learn.org/stable/documentation.html](https://scikit-learn.org/stable/documentation.html)
* **Videos**:
    * Corey Schafer's Python YouTube-Kanal: [youtube.com/@coreyms](https://www.youtube.com/channel/UCCezIgC97PvUuR4_gbFUs5g) - Ausgezeichnete Tutorials zu verschiedenen Python-Themen, einschließlich Pandas und OOP.
    * Sentdex YouTube-Kanal: [youtube.com/@sentdex](https://youtube.com/user/sentdex) - Behandelt eine breite Palette von Python-Programmierung, einschließlich Datenanalyse und maschinellem Lernen.

---

## Statistische Grundlagen & EDA (Explorative Datenanalyse) 📊

* **Bücher**:
    * "Think Stats: Exploratory Data Analysis in Python" von Allen B. Downey (Kostenlos online verfügbar unter [allendowney.github.io/ThinkStats/](https://allendowney.github.io/ThinkStats/)).
    * "Practical Statistics for Data Scientists" von Peter Bruce, Andrew Bruce & Peter Gedeck. (Deutscher Titel oft: "Statistik für Data Scientists: Praxiswissen für den Berufsalltag")
* **Online-Kurse**:
    * Khan Academy's Statistics and Probability: [khanacademy.org/math/statistics-probability](https://khanacademy.org/math/statistics-probability) (Viele Inhalte auch auf Deutsch verfügbar)
* **Artikel**:
    * "Understanding Descriptive and Inferential Statistics" - Viele gute Artikel finden sich auf Towards Data Science oder ähnlichen Blogs.
Deutsch verfügbar)
* **Videos**:
    * 3Blue1Brown Youtube-Kanal: [youtube.com/@3blue1brown](https://www.youtube.com/c/3blue1brown) - Statistische und mathematische Grundlagen gut erklärt und visualisiert.

---

## Datenvorverarbeitung (Data Preprocessing) 🛠️

* **Artikel & Dokumentation**:
    * Scikit-learn-Dokumentation zu [Preprocessing data](https://scikit-learn.org/stable/modules/preprocessing.html).
    * Artikel über Feature Scaling, Kodierung kategorialer Daten, Umgang mit fehlenden Werten (z.B. auf KDnuggets, Towards Data Science).
* **Videos**:
    * StatQuest with Josh Starmer (YouTube): Hat ausgezeichnete, leicht verständliche Videos zu verschiedenen statistischen und ML-Konzepten, einschließlich derer, die für die Vorverarbeitung relevant sind. Der [Video Index auf statquest.org](https://statquest.org/video-index/) ist sehr hilfreich.

---

## Kernalgorithmen des Maschinellen Lernens 🧠

### k-Nächste-Nachbarn (kNN)
* **Videos**:
    * StatQuest: "k-nearest neighbors (kNN) clearly explained" [Link](https://www.youtube.com/watch?v=HVXime0nQeI)
* **Artikel**:
    * "An Introduction to K-Nearest Neighbours Algorithm" auf Towards Data Science: [Link](https://towardsdatascience.com/an-introduction-to-k-nearest-neighbours-algorithm-3ddc99883acd)

### Lineare & Polynomiale Regression
* **Bücher**:
    * Kapitel in "An Introduction to Statistical Learning."
* **Videos**:
    * StatQuest: "Linear Regression, Clearly Explained" [Link](https://www.youtube.com/watch?v=7ArmBVF2dCs)
    * Khan Academy Videos zur linearen Regression.

### Logistische Regression
* **Videos**:
    * StatQuest: "Logistic Regression" ([Playlist](https://www.youtube.com/playlist?list=PLblh5JKOoLUKxzEP5HA2d-Li7IJkHfXSe))
* **Artikel**:
    * "Logistic Regression: Detailed Overview" (z.B. "Logistic Regression - Explained" auf Towards Data Science): [Link](https://towardsdatascience.com/logistic-regression-explained-593e9ddb7c6c)

### Entscheidungsbäume & Ensemble-Methoden (Random Forest, AdaBoost)
* **Bücher**:
    * Kapitel in "Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow." (Deutscher Titel oft: "Praxiseinstieg Machine Learning mit Scikit-Learn, Keras & TensorFlow")
* **Videos**:
    * StatQuest: Decision Trees (z.B. [Playlist](https://www.youtube.com/playlist?list=PLblh5JKOoLUKAtDViTvRGFpphEc24M-QH)), Random Forests, AdaBoost, Gradient Boost (XGBoost) - alle haben eigene Videos.
* **Artikel**:
    * "Understanding Random Forest" (z.B. "Random Forest, Explained: A Visual Guide with Code Examples" auf Towards Data Science): [Link](https://towardsdatascience.com/random-forest-explained-a-visual-guide-with-code-examples-9f736a6e1b3c)
    * "A Gentle Introduction to AdaBoost" auf Machine Learning Plus: [Link](https://www.machinelearningplus.com/machine-learning/introduction-to-adaboost/)

### Support Vector Machines (SVM)
* **Videos**:
    * StatQuest: "Support Vector Machines (SVMs), Clearly Explained" [Link auf statquest.org](https://statquest.org/support-vector-machines-clearly-explained/)
* **Artikel**:
    * "Understanding Support Vector Machine (SVM) algorithm" auf Towards Data Science.  (z.B. "Support Vector Machine (SVM) Explained" auf Towards Data Science): [Link](https://towardsdatascience.com/support-vector-machine-svm-explained-58e59708cae3)

### Naive Bayes
* **Videos**:
    * StatQuest: "Naive Bayes, Clearly Explained"
* **Artikel**:
    * "Naive Bayes Classifier Explained" auf Towards Data Science: [Link](https://towardsdatascience.com/naive-bayes-classifier-explained-50f9723571ed)

---

## Modellbewertung & -verbesserung ⚙️✨

* **Dokumentation & Artikel**:
    * Scikit-learn-Dokumentation zu [Model evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html) (ROC, AUC, etc.).
    * Scikit-learn-Dokumentation zu [Tuning the hyper-parameters of an estimator](https://scikit-learn.org/stable/modules/grid_search.html).
    * "Understanding AUC - ROC Curve" (z.B. "Understanding AUC Scores in Depth: What's the Point?" auf Towards Data Science): [Link](https://towardsdatascience.com/understanding-auc-scores-in-depth-whats-the-point-5f2505eb499f)
    * "Hyperparameter Tuning Techniques in Machine Learning" (z.B. "Hyperparameter tuning in Python" auf Towards Data Science): [Link](https://towardsdatascience.com/hyperparameter-tuning-in-python-21a76794a1f7)
    * "A Short Introduction to the Curse of Dimensionality" von Lilian Weng (Besuchen Sie Lilian Weng's Blog [lil-log](https://lilianweng.github.io/lil-log/) und suchen Sie nach "Curse of Dimensionality", z.B. im Kontext von Policy Gradients [hier](https://lilianweng.github.io/posts/2018-04-08-policy-gradient/)).
* **Videos**:
    * StatQuest: "ROC and AUC, Clearly Explained"
    * Videos zu Kreuzvalidierung und Hyperparameter-Tuning (z.B. von Sentdex oder Krish Naik auf YouTube).

### Modellinterpretation (SHAP)
* **Bücher/Paper**:
    * Das ursprüngliche SHAP-Paper: "A Unified Approach to Interpreting Model Predictions" von Lundberg und Lee [Link](https://proceedings.neurips.cc/paper/2017/file/8a20a8621978632d76c43dfd28b67767-Paper.pdf).
* **Dokumentation & Artikel**:
    * SHAP GitHub Repository und Dokumentation: [github.com/slundberg/shap](https://github.com/slundberg/shap)
    * "Explain Your Model with SHAP Values" auf Towards Data Science (z.B. "SHapley Additive exPlanations or SHAP : What is it ?" auf DataScientest): [Link](https://datascientest.com/en/shap-what-is-it).

---

## Deep Learning 💡

* **Bücher**:
    * "Deep Learning" von Ian Goodfellow, Yoshua Bengio und Aaron Courville (Das "Deep Learning Buch" - kostenlos online unter [www.deeplearningbook.org](https://www.deeplearningbook.org/)).
    * "Deep Learning with Python" von François Chollet (Entwickler von Keras). (Deutscher Titel oft: "Deep Learning mit Python und Keras")
* **Videos**:
    * 3Blue1Brown YouTube-Kanal: Neural Networks series [Playlist](youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) auf dem Kanal [youtube.com/@3blue1brown](http://googleusercontent.com/youtube.com/c/3blue1brown) für ein intuitives Verständnis.
    * Lex Fridman Podcast: [Podcast Playlist](youtube.com/playlist?list=PLrAXtmErZgOdP_8GztsuKi9nrraNbKKp4) - Interviews mit führenden KI-Forschern (eher für breiteren Kontext und Inspiration).
    * Two Minute Papers YouTube-Kanal: [youtube.com/@TwoMinutePapers](http://googleusercontent.com/youtube.com/user/TwoMinutePapers) - Fasst spannende neue KI-Forschungsarbeiten zusammen.

### PyTorch
* **Dokumentation & Tutorials**:
    * Offizielle PyTorch-Tutorials: [pytorch.org/tutorials/](https://pytorch.org/tutorials/)

### Convolutional Neural Networks (CNN)
* **Artikel**:
    * "A Comprehensive Guide to Convolutional Neural Networks" auf Towards Data Science [PDF Link](https://ise.ncsu.edu/wp-content/uploads/sites/9/2022/08/A-Comprehensive-Guide-to-Convolutional-Neural-Networks-%E2%80%94-the-ELI5-way-_-by-Sumit-Saha-_-Towards-Data-Science.pdf).
    * Stanford CS231n: Convolutional Neural Networks for Visual Recognition (Kurs-Website 2017 mit Notizen): [cs231n.stanford.edu/2017/](https://cs231n.stanford.edu/2017/)
* **Videos**:
    * Viele Videos auf YouTube, die CNN-Architekturen erklären (z.B. LeNet, AlexNet, VGG, ResNet).

### Representation Learning (Autoencoder, PCA, t-SNE)
* **Artikel**:
    * "Understanding Autoencoders" (z.B. "Autoencoders and the Denoising Feature" auf Towards Data Science): [Link](https://towardsdatascience.com/autoencoders-and-the-denoising-feature-from-theory-to-practice-db7f7ad8fc78).
    * "Principal Component Analysis (PCA) Explained" auf Towards Data Science: [Link](https://towardsdatascience.com/principal-component-analysis-pca-8133b02f11bd) (Hauptkomponentenanalyse auf Deutsch).
    * "How to Use t-SNE Effectively" auf Distill.pub: [Link](https://distill.pub/2016/misread-tsne/)
* **Videos**:
    * StatQuest: "Principal Component Analysis (PCA), Step-by-Step"
    * StatQuest: "t-SNE, Clearly Explained"

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

## Tool- und Software-Empfehlungen 🛠️💻

Hier ist eine Liste empfohlener Tools und Software, die im Bereich Full-Stack Machine Learning häufig verwendet werden und für Ihre Lernreise nützlich sein können:

### Programmiersprachen & Kernbibliotheken

* **Python**: Die dominierende Sprache im Bereich Machine Learning und Data Science.
    * **NumPy**: Für numerische Berechnungen, insbesondere Array-Operationen.
    * **Pandas**: Für Datenmanipulation und -analyse (Stichwort: DataFrames).
    * **Scikit-learn**: Umfassende Bibliothek für klassisches Machine Learning (Klassifikation, Regression, Clustering, Dimensionsreduktion, Modellselektion, Preprocessing).
    * **Statsmodels**: Für statistische Modellierung, Tests und Datenexploration.
* **R**: Eine weitere beliebte Sprache für statistische Analysen und Datenvisualisierung.
* **SQL**: Essentiell für die Arbeit mit relationalen Datenbanken und Datenabfragen.

### Deep Learning Frameworks

* **TensorFlow (mit Keras)**: Ein umfangreiches Open-Source-Framework für Machine Learning und insbesondere Deep Learning, entwickelt von Google. Keras dient als benutzerfreundliche High-Level-API.
* **PyTorch**: Ein populäres Open-Source-Framework für Deep Learning, bekannt für seine Flexibilität und Python-freundliche Natur, entwickelt von Facebooks KI-Forschungslabor.

### IDEs (Integrierte Entwicklungsumgebungen) und Code-Editoren

* **Visual Studio Code (VS Code)**: Ein sehr beliebter, kostenloser und erweiterbarer Code-Editor mit exzellenter Python- und Jupyter-Unterstützung.
* **JupyterLab / Jupyter Notebook**: Interaktive, webbasierte Umgebungen, ideal für explorative Datenanalyse, Visualisierungen und das Teilen von Code. (Wie in diesem Kurs verwendet)
* **PyCharm**: Eine leistungsstarke IDE speziell für Python, mit einer kostenlosen Community-Version und einer kostenpflichtigen Professional-Version.
* **Google Colaboratory (Colab)**: Kostenlose Jupyter-Notebook-Umgebung, die in der Cloud läuft und Zugriff auf GPUs/TPUs bietet, ideal für Deep Learning Experimente.

### Versionskontrolle

* **Git**: Das Standard-System für verteilte Versionskontrolle, unerlässlich für die Zusammenarbeit und das Nachverfolgen von Code-Änderungen.
* **GitHub / GitLab / Bitbucket**: Web-basierte Plattformen für das Hosting von Git-Repositories, die zusätzliche Funktionen für Kollaboration, Issue-Tracking und CI/CD bieten.

### Datenvisualisierung

* **Matplotlib**: Eine grundlegende Bibliothek zur Erstellung statischer, animierter und interaktiver Visualisierungen in Python.
* **Seaborn**: Baut auf Matplotlib auf und bietet eine High-Level-Schnittstelle für ansprechendere und informativere statistische Grafiken.
* **Plotly / Dash**: Für interaktive Web-basierte Visualisierungen und Dashboards. Dash ist ein Framework zum Erstellen von Analyseanwendungen mit Python.
* **Tableau / Microsoft Power BI**: Führende Business-Intelligence-Tools für fortgeschrittene Datenvisualisierung und Dashboarding (eher für BI-Analysten, aber nützlich zu kennen).

### MLOps und Deployment

* **Docker**: Zum Erstellen, Verteilen und Ausführen von Anwendungen in Containern, um Konsistenz über verschiedene Umgebungen hinweg sicherzustellen.
* **Kubernetes (K8s)**: Ein System zur Automatisierung der Bereitstellung, Skalierung und Verwaltung von containerisierten Anwendungen.
* **MLflow**: Eine Open-Source-Plattform zur Verwaltung des gesamten Machine-Learning-Lebenszyklus, einschließlich Experiment-Tracking, Code-Packaging, Modell-Deployment und Modell-Registry.
* **DVC (Data Version Control)**: Ein Tool zur Versionskontrolle von Daten und Machine-Learning-Modellen, oft in Verbindung mit Git verwendet.
* **FastAPI / Flask**: Python-Webframeworks, die häufig zum Erstellen von APIs für das Deployment von ML-Modellen verwendet werden.
* **BentoML**: Ein Framework zum Erstellen produktionsreifer KI-Anwendungen.

### Cloud-Plattformen

Viele Cloud-Anbieter bieten umfassende Suiten für Data Science, Machine Learning und MLOps:

* **Amazon Web Services (AWS)**:
    * **Amazon SageMaker**: Vollständig verwaltete Plattform für den gesamten ML-Workflow.
    * **EC2** (Compute), **S3** (Storage), diverse Datenbankdienste.
* **Google Cloud Platform (GCP)**:
    * **Vertex AI**: Einheitliche ML-Plattform.
    * **Google Compute Engine**, **Cloud Storage**, **BigQuery**.
* **Microsoft Azure**:
    * **Azure Machine Learning**: Umfassender Dienst für ML-Entwicklung und -Deployment.
    * **Virtual Machines**, **Blob Storage**, diverse Datenbankdienste.

### Datenbanktechnologien

* **Relationale Datenbanken (SQL)**: z.B. PostgreSQL, MySQL, SQLite.
* **NoSQL-Datenbanken**: z.B. MongoDB (dokumentenorientiert), Redis (Key-Value), Cassandra (spaltenorientiert) – je nach Anwendungsfall.

Diese Liste ist nicht abschließend, aber sie deckt viele der wichtigsten Werkzeuge ab, denen Sie in der Praxis begegnen werden. Die Auswahl der richtigen Tools hängt oft von den spezifischen Anforderungen des Projekts, des Teams und der Organisation ab.

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

---

## Konferenzen (Auswahl) 📅🏛️

Konferenzen sind eine großartige Möglichkeit, sich über die neuesten Entwicklungen zu informieren und Kontakte zu knüpfen (viele bieten auch Online-Zugang oder Aufzeichnungen):

* **International**:
    * **NeurIPS** (Conference on Neural Information Processing Systems): [nips.cc](https://nips.cc/)
    * **ICML** (International Conference on Machine Learning): [icml.cc](https://icml.cc/)
    * **CVPR** (Conference on Computer Vision and Pattern Recognition): [cvpr.thecvf.com](http://cvpr.thecvf.com/)
    * **ACL** (Annual Meeting of the Association for Computational Linguistics): [aclweb.org](https://www.aclweb.org/portal/)
    * **KDD** (ACM SIGKDD Conference on Knowledge Discovery and Data Mining): [kdd.org](https://www.kdd.org/)
* **Europa / Deutschland (Beispiele)**:
    * **PyData Conferences**: [pydata.org](https://pydata.org/) (weltweit, auch in Europa, z.B. Berlin, Amsterdam)
    * **ODSC (Open Data Science Conference) Europe**: [odsc.com/europe/](https://odsc.com/europe/)
    * **Data Natives**: [datanatives.io](https://datanatives.io/) (oft in Berlin)
    * Achten Sie auf lokale Universitäts-Workshops und Industrieveranstaltungen.

---
