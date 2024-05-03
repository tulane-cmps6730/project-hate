# Hate Speech on Social Media
### Abstract
Hate speech has become of concern online as criminally threatening, as a possible predictor of violent activity, and as an important con- tributor to radicalization. However, most studies have focused on explicit hate speech which clearly states the bias and intentions of the speaker, and have focused on identifying hate speech based on key derogatory phrases. Study is needed of implicit hate speech, which requires semantic and contextual understanding of language, making it more difficult for computers to identify and making it less likely for a speaker to be censored. Past study has classified hate speech in a variety of ways, including as implicit or explicit. This study ac- knowledges that hate speech may exist on a gradient from implicit to explicit and compares performance of a simple classification model to performance of two different regression models to determine whether hate speech can be understood as a gradient rather than discrete classes, with the possibility that some text’s position on that gradi- ent is a function of speech in context. This study finds that the model of text that considers both speech and context performs better than a simple linear regression and that its hidden representations of speech and context converge to be of equal magnitude and opposite signage. Overall, this study finds that the simple neural network classifier out performs the linear regressions, suggesting that hate speech is indeed discrete and unordered in categorization.

###Web App (in /nlp folder)
- A simple web UI using Flask to support a demo of the project
- A command-line interface to support running different stages of the project's pipeline
- The ability to easily reproduce your work on another machine by using virtualenv and providing access to external data sources.

### Contents

- [docs](docs): template to create slides for project presentations
- [nlp](nlp): Python project code--can be downloaded and deployed to a web app.
- [notebooks](notebooks): Jupyter notebooks for project development and experimentation. Main Project is in project_final.ipynb.
- [report](report): LaTeX report
- [tests](tests): unit tests for project code. See demo video uploaded here including tests based on things I thought up, text taken from the NationalFascism subreddit, and an angry email I got last year when working in politics.
