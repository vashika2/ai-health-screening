import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()  # Reads your .env file

# Initialize Groq client with your free API key
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "llama-3.3-70b-versatile"  # Free Llama 3 model

def generate_tb_report(prediction: dict, patient_info: dict = None) -> str:
    """
    Send TB prediction results to Llama 3 and get back a
    plain-English report a rural health worker can understand.
    """
    patient_context = ""
    if patient_info:
        patient_context = f"""
Patient Information:
- Age: {patient_info.get('age', 'Not provided')}
- Symptoms: {patient_info.get('symptoms', 'Not provided')}
- Duration of symptoms: {patient_info.get('duration', 'Not provided')}
        """

    prompt = f"""
You are a medical AI assistant helping health workers at rural clinics in India.
Write a clear, simple preliminary screening report based on this chest X-ray AI analysis:

Result: {prediction['label']}
Confidence: {prediction['confidence']}%
Severity: {prediction['severity']}
Urgent Referral Needed: {prediction['needs_referral']}
{patient_context}

Write the report with exactly these 4 sections:
1. FINDINGS
2. INTERPRETATION
3. RECOMMENDED NEXT STEPS
4. IMPORTANT DISCLAIMER

Rules:
- Under 200 words total
- Simple language any non-specialist can understand
- Be direct and actionable
- If referral is needed, say so clearly and urgently
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
        temperature=0.3  # Low temperature = more consistent, factual output
    )

    return response.choices[0].message.content

def generate_dr_report(prediction: dict, patient_info: dict = None) -> str:
    """
    Same idea for Diabetic Retinopathy — grade 0-4 gets turned
    into a clear report with vision risk and next steps.
    """
    patient_context = ""
    if patient_info:
        patient_context = f"""
Patient Information:
- Age: {patient_info.get('age', 'Not provided')}
- Diabetes duration: {patient_info.get('diabetes_years', 'Not provided')} years
- Last HbA1c: {patient_info.get('hba1c', 'Not provided')}
        """

    prompt = f"""
You are a medical AI assistant helping health workers at rural clinics in India.
Write a clear, simple screening report for this Diabetic Retinopathy result:

DR Grade: {prediction['grade']} out of 4
Classification: {prediction['label']}
Confidence: {prediction['confidence']}%
Urgent Referral Needed: {prediction['needs_referral']}
All Grade Probabilities: {prediction['all_grades']}
{patient_context}

Write the report with exactly these 5 sections:
1. FINDINGS
2. SEVERITY EXPLANATION (explain what Grade {prediction['grade']} means for the patient)
3. RISK TO VISION
4. RECOMMENDED NEXT STEPS
5. IMPORTANT DISCLAIMER

Rules:
- Under 200 words total
- Simple language only — patient and health worker should both understand
- If Grade 3 or 4, emphasize urgency very strongly
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
        temperature=0.3
    )

    return response.choices[0].message.content