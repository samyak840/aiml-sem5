from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

OUT = '/Users/samyakbelida/Desktop/LoanLens/LoanLens_Presentation_Guide.docx'
doc = Document()
sec = doc.sections[0]
sec.top_margin = Inches(0.65)
sec.bottom_margin = Inches(0.65)
sec.left_margin = Inches(0.8)
sec.right_margin = Inches(0.8)
styles = doc.styles
styles['Normal'].font.name = 'Aptos'
styles['Normal']._element.rPr.rFonts.set(qn('w:eastAsia'), 'Aptos')
styles['Normal'].font.size = Pt(10.5)
styles['Normal'].paragraph_format.space_after = Pt(6)
styles['Normal'].paragraph_format.line_spacing = 1.08
for name, size in [('Title', 25), ('Heading 1', 16), ('Heading 2', 12.5), ('Heading 3', 11)]:
    st = styles[name]
    st.font.name = 'Aptos Display'
    st._element.rPr.rFonts.set(qn('w:eastAsia'), 'Aptos Display')
    st.font.size = Pt(size)
    st.font.bold = True
    st.font.color.rgb = RGBColor(0, 0, 0)
title_ppr = styles['Title']._element.get_or_add_pPr()
title_border = title_ppr.find(qn('w:pBdr'))
if title_border is not None:
    title_ppr.remove(title_border)

def shade(cell, fill):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:fill'), fill)
    tcPr.append(shd)

def borders(table):
    tblPr = table._tbl.tblPr
    b = OxmlElement('w:tblBorders')
    for edge in ('top', 'left', 'bottom', 'right', 'insideH', 'insideV'):
        el = OxmlElement('w:' + edge)
        el.set(qn('w:val'), 'single')
        el.set(qn('w:sz'), '5')
        el.set(qn('w:space'), '0')
        el.set(qn('w:color'), 'D9D9D9')
        b.append(el)
    tblPr.append(b)

def add_table(headers, rows):
    t = doc.add_table(rows=1, cols=len(headers))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    borders(t)
    for i, text in enumerate(headers):
        c = t.rows[0].cells[i]
        c.text = text
        shade(c, '163A5F')
        c.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
        for r in c.paragraphs[0].runs:
            r.bold = True
            r.font.color.rgb = RGBColor(255, 255, 255)
            r.font.size = Pt(9)
    for ri, row in enumerate(rows):
        cells = t.add_row().cells
        for i, text in enumerate(row):
            cells[i].text = str(text)
            cells[i].vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            if ri % 2:
                shade(cells[i], 'F2F6F8')
            for r in cells[i].paragraphs[0].runs:
                r.font.size = Pt(9)
    doc.add_paragraph()

def bullet(text):
    p = doc.add_paragraph(style='List Bullet')
    p.paragraph_format.space_after = Pt(2)
    p.add_run(text)

def number(text):
    p = doc.add_paragraph(style='List Number')
    p.paragraph_format.space_after = Pt(3)
    p.add_run(text)

def say(text):
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Inches(0.25)
    p.paragraph_format.space_after = Pt(7)
    r = p.add_run(text)
    r.italic = True
    r.font.color.rgb = RGBColor.from_string('163A5F')

def h1(text):
    doc.add_heading(text, level=1)

def h2(text):
    doc.add_heading(text, level=2)

# Cover page
p = doc.add_paragraph(style='Title')
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.add_run('LoanLens Presentation Guide')
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = p.add_run('Loan approval prediction and financial risk analysis')
r.bold = True
r.font.size = Pt(14)
r.font.color.rgb = RGBColor.from_string('163A5F')
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.add_run('Complete speaking script and live demonstration plan')
say('Purpose: present the project clearly, demonstrate the complete workflow, explain the model score correctly, and answer common evaluation questions.')
add_table(['Project fact', 'What to remember'], [
    ('Application type', 'Local Streamlit decision-support prototype'),
    ('Dataset', '20,000 rows from data/Loan.csv'),
    ('Target', 'LoanApproved: approved or rejected'),
    ('Selected model', 'Logistic Regression, selected by highest ROC-AUC'),
    ('Held-out test accuracy', '96.13%'),
    ('Held-out test ROC-AUC', '99.48%'),
])
say('Important: these are dataset-specific benchmark results. LoanLens is not a production lending system and should not be used as the sole basis for a real lending decision.')
doc.add_page_break()

# Opening and architecture
h1('1 Presentation opening')
doc.add_paragraph('Start with the problem, then introduce the solution. Do not begin with code or model names.')
say('Good morning. My project is LoanLens, a loan approval and financial risk analysis application. The purpose of the project is to show how machine learning can support the review of a hypothetical loan application while also presenting transparent affordability and risk indicators.')
say('The application accepts applicant and loan information, predicts the probability of approval or rejection, calculates EMI and debt-to-income indicators, identifies unusual profiles for review, and stores completed assessments for later analysis.')
h2('Project objectives')
for x in ['Build a complete machine-learning workflow for loan approval prediction.', 'Compare multiple classification models and select the strongest model using ROC-AUC.', 'Provide understandable prediction factors instead of showing only a final label.', 'Separate statistical prediction from rule-based financial risk analysis.', 'Store and export assessment history for review.', 'Demonstrate responsible use and explain project limitations.']:
    bullet(x)
h2('One sentence summary')
say('LoanLens is a decision-support dashboard that combines a trained loan approval classifier with transparent affordability formulas, anomaly monitoring, explainability, and local history management.')
h1('2 Project architecture')
add_table(['Layer', 'Project files', 'Role'], [
    ('User interface', 'app.py, assets/style.css', 'Streamlit pages, forms, charts, and results'),
    ('Data preparation', 'src/data_preprocessing.py, config.py', 'Load data, select features, impute, scale, and encode'),
    ('Model training', 'src/train_model.py', 'Train four classifiers and select the best one'),
    ('Prediction', 'loan_model.py, src/prediction.py', 'Validate input and generate approval probabilities'),
    ('Risk analysis', 'src/risk_analysis.py', 'Calculate EMI, DTI, LTI, payment burden, and risk score'),
    ('Explainability', 'src/explainability.py, loan_model.py', 'Show local factor comparisons and feature importance'),
    ('Persistence', 'src/database.py, data/loaniq.db', 'Save, filter, view, export, and delete assessments'),
    ('Evaluation', 'src/evaluation.py, tests/', 'Charts and automated checks for model behavior'),
])
h2('End-to-end data flow')
say('Application form → input validation → model prediction → risk and affordability calculations → explanation and anomaly review → optional SQLite save → dashboard and history reporting.')
doc.add_page_break()

# Live demo
h1('3 Live demonstration steps')
doc.add_paragraph('Use this order during the live demo. It tells a complete story and ensures that the History and Dashboard pages contain saved results.')
h2('Step 1 Start the application')
doc.add_paragraph('From the project folder, run:')
p = doc.add_paragraph(style='No Spacing')
r = p.add_run('cd /Users/samyakbelida/Desktop/LoanLens\npython3 -m venv .venv\nsource .venv/bin/activate\npython3 -m pip install -r requirements.txt\npython3 -m streamlit run app.py')
r.font.name = 'Courier New'
r.font.size = Pt(9)
doc.add_paragraph('Open the local Streamlit address, normally http://localhost:8501.')
say('The application is running locally. On startup it loads the supplied loan dataset and saved model artifacts. If artifacts are missing or outdated, it trains the comparison models automatically.')
h2('Step 2 Show the Dashboard')
for x in ['Select Dashboard from the sidebar.', 'Point out the dataset applications, model inputs, and saved assessments.', 'Show the approval, loan amount, income, credit score, and employment charts.']:
    number(x)
say('This dashboard gives a portfolio-level view. The historical charts come from the source dataset, while the saved assessment charts are populated by applications evaluated during the demo.')
h2('Step 3 Assess a normal hypothetical applicant')
for x in ['Select New Loan Application.', 'Leave the dataset-based default values in place for the first run.', 'Enter a non-zero value for Other monthly expenses, such as 1,000.', 'Click Assess application.']:
    number(x)
say('The form contains income, assets, loan request, existing obligations, credit history, housing, and monthly expenses. The application validates inputs before running the prediction.')
h2('Step 4 Explain the result')
for x in ['Point out the indicative approval or rejection result.', 'Show approval probability, rejection probability, financial risk score, and risk band.', 'Explain Positive indicators and Potential concerns as local model comparisons, not causal explanations.']:
    number(x)
say('The approval probability comes from the selected classifier. The financial risk score is calculated separately from financial indicators. Keeping these outputs separate makes the result easier to interpret.')
h2('Step 5 Demonstrate financial analysis')
for x in ['Show estimated EMI, existing DTI, payment burden, loan-to-income, disposable income, and anomaly status.', 'Increase Other monthly expenses and submit again.', 'Show that disposable income decreases and affordability may change.']:
    number(x)
say('The EMI uses the standard amortization formula. Entered non-debt expenses are used for affordability only; they are not sent to the prediction model.')

# Comparison and pages
doc.add_page_break()
h1('4 Strong comparison demonstration')
doc.add_paragraph('A two-scenario comparison demonstrates that the outputs respond to different financial profiles.')
add_table(['Scenario', 'Example changes', 'What to discuss'], [
    ('Scenario A: stronger profile', 'Higher credit score, lower debt payments, moderate loan amount, lower expenses', 'Show approval probability, lower financial pressure, and disposable income'),
    ('Scenario B: higher-risk profile', 'Lower credit score, higher loan amount, higher debt payments, high utilization, defaults, higher expenses', 'Show higher risk indicators, lower headroom, and possible anomaly review'),
])
doc.add_paragraph('Do not promise an exact result before submitting because the model evaluates all input features together. Explain the direction of indicators rather than claiming one field alone caused the decision.')
h2('Remaining pages')
for x in ['Loan Prediction: focuses on model output, probabilities, risk score, risk category, and explanation.', 'Financial Risk Analysis: focuses on rule-based affordability, risk components, anomaly status, and saved risk distribution.', 'Application History: demonstrates filters, lookup by ID, detailed record view, and CSV export.', 'Model Performance: demonstrates comparison metrics, confusion matrix, ROC curve, classification report, and feature importance.', 'About Project: reinforces assumptions, privacy limitations, responsible use, and the difference between prediction and risk score.']:
    bullet(x)
h2('Save the assessment')
for x in ['Return to New Loan Application.', 'Click Save assessment to history.', 'Note the generated assessment ID.']:
    number(x)
say('The record is saved locally in SQLite. This prototype does not collect an applicant name and should not be used with real personal or financial data.')

# Technical and metrics
doc.add_page_break()
h1('5 Technical explanation')
h2('Dataset and features')
doc.add_paragraph('The main dataset is data/Loan.csv with 20,000 rows. The target is LoanApproved, where 1 means approved and 0 means rejected. The model uses 31 numeric and categorical inputs covering income, assets, credit, debt, employment, education, housing, and loan purpose.')
doc.add_paragraph('LoanApproved and RiskScore are excluded because they are the target and an outcome-like field. ApplicationDate is excluded because time drift is not modeled. Age and MaritalStatus are excluded from automated eligibility prediction as sensitive applicant attributes.')
h2('Preprocessing')
for x in ['Numeric missing values are filled with training-set medians.', 'Categorical missing values are filled with the most frequent training value.', 'Numeric variables are standardized.', 'Categorical variables are one-hot encoded.', 'Class balancing is enabled for the main classification models.', 'Preprocessing is fitted inside each pipeline to avoid test-set leakage.']:
    bullet(x)
h2('Models compared')
add_table(['Model', 'Test accuracy', 'Test ROC-AUC', 'Role'], [
    ('Logistic Regression', '96.13%', '99.48%', 'Selected model'),
    ('Gradient Boosting', '95.90%', '99.07%', 'Second highest ROC-AUC'),
    ('Random Forest', '91.73%', '97.69%', 'Saved for feature importance'),
    ('Decision Tree', '88.23%', '93.73%', 'Comparison baseline'),
])
doc.add_paragraph('The model is selected by highest ROC-AUC, not accuracy alone. The saved selected model is Logistic Regression.')

doc.add_page_break()
h1('6 Accuracy and ROC AUC explanation')
h2('What to tell the teacher')
say('The selected Logistic Regression model achieved 96.13% accuracy on the held-out test set of 4,000 applications. It achieved a ROC-AUC of 99.48%. The model was selected by comparing four classifiers using held-out evaluation metrics.')
h2('How accuracy is calculated')
doc.add_paragraph('Accuracy is the proportion of all predictions that are correct:')
say('Accuracy = (correct approvals + correct rejections) ÷ total test applications')
doc.add_paragraph('The selected model correctly classified 2,914 rejected cases and 931 approved cases. It made 130 false approvals and missed 25 approved cases. Therefore, 3,845 of 4,000 test applications were correct, giving 96.13% accuracy.')
h2('How ROC-AUC is measured')
doc.add_paragraph('The model outputs an approval probability. The evaluation tests many probability thresholds. At each threshold it calculates true positive rate, which measures approved applications correctly identified, and false positive rate, which measures rejected applications incorrectly identified as approved. These points form the ROC curve; the area under that curve is ROC-AUC.')
say('A ROC-AUC of 0.9948 means the model has approximately a 99.48% chance of ranking a randomly chosen approved application above a randomly chosen rejected application in this held-out dataset.')
doc.add_paragraph('ROC-AUC is not the same as accuracy. Accuracy uses one decision threshold, normally 50%, while ROC-AUC evaluates ranking quality across many thresholds.')
h2('Important qualification')
doc.add_paragraph('The project reports held-out test performance, not performance on an external banking dataset. Training accuracy is not currently saved in the project artifacts, so do not claim a training accuracy number unless it is recalculated separately. The honest benchmark statement is 96.13% accuracy on held-out data.')

# Financials, Q&A, checklist
doc.add_page_break()
h1('7 Financial formulas and risk score')
add_table(['Indicator', 'Formula or rule', 'Meaning'], [
    ('DTI', 'Existing monthly debt ÷ gross monthly income × 100', 'Share of income already committed to debt'),
    ('EMI', 'P × r × (1+r)^n ÷ ((1+r)^n − 1)', 'Estimated monthly payment for the proposed loan'),
    ('LTI', 'Requested loan amount ÷ annual income', 'Loan size relative to annual income'),
    ('Payment burden', '(Existing debt + proposed EMI) ÷ monthly income × 100', 'Total monthly debt-payment pressure'),
    ('Disposable income', 'Monthly income − expenses − all EMIs', 'Remaining monthly income after commitments'),
])
doc.add_paragraph('The annual interest rate is entered as a decimal fraction. For example, 0.24 means 24%. Loan duration is entered in months.')
h2('Risk categories')
add_table(['Score range', 'Category', 'Interpretation'], [
    ('0–34', 'Low', 'Lower rule-based financial pressure'),
    ('35–64', 'Medium', 'Moderate pressure requiring review'),
    ('65–100', 'High', 'Higher rule-based financial pressure'),
])
doc.add_paragraph('The risk score is a capped sum of indicators. It is not a calibrated probability of repayment, default, or fraud. The anomaly monitor identifies statistically unusual profiles for human review; it does not prove fraud.')
h1('8 Questions the teacher may ask')
qa = [
('Why Logistic Regression?', 'It had the highest ROC-AUC, 0.9948, on held-out data.'),
('Why not accuracy alone?', 'Accuracy can hide poor performance on one class, so precision, recall, F1, ROC-AUC, and the confusion matrix are also shown.'),
('What is ROC-AUC?', 'It is the area under the curve comparing true positive rate and false positive rate across many thresholds.'),
('Prediction versus risk score?', 'Prediction comes from the classifier; risk score is a separate transparent financial formula.'),
('Is it production-ready?', 'No. It lacks authentication, encryption at rest, production access controls, calibration, fairness validation, and external validation.'),
('Can it make real lending decisions?', 'No. It is a prototype and requires qualified human review.'),
('Where is data saved?', 'Assessments are stored locally in data/loaniq.db and can be exported as CSV.'),
('Why exclude Age and MaritalStatus?', 'They are sensitive attributes excluded from automated eligibility prediction in this prototype.'),
]
for q, a in qa:
    p = doc.add_paragraph()
    rr = p.add_run('Question: ' + q)
    rr.bold = True
    p2 = doc.add_paragraph('Answer: ' + a)
    p2.paragraph_format.left_indent = Inches(0.18)

doc.add_page_break()
h1('9 Final presentation checklist')
for x in ['Start the app and confirm the local URL opens.', 'Prepare a stronger-profile and a higher-risk hypothetical scenario.', 'Save at least one assessment so Dashboard and History contain live results.', 'Remember: 96.13% held-out test accuracy.', 'Remember: Logistic Regression was selected by ROC-AUC.', 'Explain ROC-AUC as ranking quality across thresholds.', 'Do not claim 96.13% accuracy on all future real-world applications.', 'Do not enter real personal or financial information.', 'Finish by stating the limitations and need for human review.']:
    bullet(x)
h2('Closing statement')
say('In conclusion, LoanLens demonstrates a complete and explainable machine-learning workflow for loan assessment. It combines prediction, financial analysis, history tracking, and model evaluation, while communicating that the output is a prototype decision-support result rather than a real lending decision.')

for section in doc.sections:
    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = footer.add_run('LoanLens Presentation Guide')
    run.font.size = Pt(8)
    run.font.color.rgb = RGBColor.from_string('6B7280')
for p in doc.paragraphs:
    if p.style.name.startswith('Heading'):
        p.paragraph_format.keep_with_next = True
doc.save(OUT)
print(OUT)
