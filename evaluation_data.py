import json

evaluation_data = [
    {
        "id": 1,
        "question": "What is health insurance?",
        "ground_truth": "Health insurance is a contract between an insurer and a policyholder that covers medical expenses and treatments in exchange for regular premium payments.",
        "contexts": [
            "Health insurance provides coverage for doctor visits, hospital stays, prescriptions, and preventive care.",
            "Health insurance plans require regular premium payments and may include deductibles and co-pays."
        ]
    },
    {
        "id": 2,
        "question": "What are the documents required to file a claim?",
        "ground_truth": "Documents required include a filled claim form, ID proof, discharge summary, medical prescriptions, investigation reports, hospital bills, and pharmacy receipts.",
        "contexts": [
            "Common documents include claim form, ID, proof of loss, medical bills, receipts, and insurance policy.",
            "Additional supporting documents may include witness statements, repair estimates, and correspondence."
        ]
    },
    {
        "id": 3,
        "question": "What are the diseases on eye?",
        "ground_truth": "Entropion: eyelid turns inward; Ectropion: eyelid turns outward; Blepharoptosis: upper eyelid droops; Pterygium: growth on cornea; Cataract: clouding of lens affecting vision.",
        "contexts": [
            "Entropion causes eyelashes to rub the cornea. Ectropion exposes the eyelid margin. Blepharoptosis is drooping eyelid. Pterygium is a growth on the cornea. Cataract is lens clouding affecting vision."
        ]
    },
    {
        "id": 4,
        "question": "What is the premium in health insurance?",
        "ground_truth": "Premium is the regular payment made by the policyholder to the insurance company to maintain coverage.",
        "contexts": [
            "Premiums are paid monthly, quarterly, or annually for health insurance coverage.",
            "The premium amount depends on age, health status, and coverage level."
        ]
    },
    {
        "id": 5,
        "question": "What is a deductible in insurance?",
        "ground_truth": "A deductible is the amount the insured must pay out-of-pocket before the insurance coverage starts.",
        "contexts": [
            "Deductibles apply to health, auto, and property insurance claims.",
            "Higher deductibles usually result in lower premium payments."
        ]
    },
    {
        "id": 6,
        "question": "What are the types of health insurance plans?",
        "ground_truth": "Health insurance plans include individual plans, group plans, government plans like Medicare and Medicaid, and short-term plans.",
        "contexts": [
            "Individual plans are purchased directly by a person or family.",
            "Group plans are provided by employers to employees and their families."
        ]
    },
    {
        "id": 7,
        "question": "What is a claim form?",
        "ground_truth": "A claim form is an official document that must be filled out to request payment or benefits from an insurance company.",
        "contexts": [
            "Claim forms collect information about the insured, the incident, and the losses or damages.",
            "It is the first step in filing an insurance claim."
        ]
    },
    {
        "id": 8,
        "question": "What is a police report required for?",
        "ground_truth": "A police report is required to document incidents like theft, accidents, or injuries for filing a claim.",
        "contexts": [
            "Police reports serve as official evidence of an incident.",
            "Insurance companies often require a police report before processing claims."
        ]
    },
    {
        "id": 9,
        "question": "What is a medical record in claims?",
        "ground_truth": "Medical records are documents like doctor notes, prescriptions, and hospital bills used to support health insurance claims.",
        "contexts": [
            "They provide proof of treatment and expenses incurred.",
            "Medical records help the insurer verify the claim's legitimacy."
        ]
    },
    {
        "id": 10,
        "question": "What are receipts used for in a claim?",
        "ground_truth": "Receipts are proof of payment for medical treatment, repair, or other services relevant to a claim.",
        "contexts": [
            "Receipts should include date, amount, and service provider details.",
            "They help insurers verify the expenses claimed."
        ]
    },
    {
        "id": 11,
        "question": "What is an insurance policy document?",
        "ground_truth": "An insurance policy document outlines the terms, coverage, exclusions, and premiums of the insurance agreement.",
        "contexts": [
            "It serves as the legal contract between the insurer and policyholder.",
            "Policy documents include the policy number, coverage details, and terms and conditions."
        ]
    },
    {
        "id": 12,
        "question": "What is proof of loss?",
        "ground_truth": "Proof of loss is documentation demonstrating that a financial loss occurred, often required for insurance claims.",
        "contexts": [
            "It can include photos, videos, invoices, or reports.",
            "Proof of loss helps the insurer assess the claim."
        ]
    },
    {
        "id": 13,
        "question": "What is workers' compensation?",
        "ground_truth": "Workers' compensation provides wage replacement and medical benefits to employees injured during employment.",
        "contexts": [
            "It covers medical treatment, rehabilitation, and lost wages.",
            "Employees must file a claim to receive workers' compensation benefits."
        ]
    },
    {
        "id": 14,
        "question": "What is a witness statement?",
        "ground_truth": "A witness statement is a written account from someone who saw or heard events relevant to a claim.",
        "contexts": [
            "It can support personal injury, accident, or property damage claims.",
            "Witness statements increase the credibility of the claim."
        ]
    },
    {
        "id": 15,
        "question": "What is an affidavit?",
        "ground_truth": "An affidavit is a written statement sworn under oath, often required to provide additional information for a claim.",
        "contexts": [
            "It can be used to certify facts, incidents, or identity.",
            "Affidavits are legal documents that may be required by courts or insurers."
        ]
    },
    {
        "id": 16,
        "question": "What is co-insurance?",
        "ground_truth": "Co-insurance is the percentage of costs the insured must pay for covered services, with the insurer paying the remaining percentage.",
        "contexts": [
            "Co-insurance is common in health insurance after the deductible is met.",
            "It reduces the insurer's risk while sharing costs with the policyholder."
        ]
    },
    {
        "id": 17,
        "question": "What is preventive care?",
        "ground_truth": "Preventive care includes medical services like vaccinations, screenings, and check-ups to prevent illness.",
        "contexts": [
            "Preventive care is often covered under health insurance plans.",
            "It helps maintain health and detect issues early."
        ]
    },
    {
        "id": 18,
        "question": "What is a network in health insurance?",
        "ground_truth": "A network is a list of healthcare providers contracted with the insurer, offering discounted rates to policyholders.",
        "contexts": [
            "Using in-network providers reduces out-of-pocket costs.",
            "Out-of-network providers may have limited or no coverage."
        ]
    },
    {
        "id": 19,
        "question": "What is Medicare?",
        "ground_truth": "Medicare is a government program that provides health coverage to individuals 65+ and certain disabled persons.",
        "contexts": [
            "Medicare covers hospital, medical, and prescription services.",
            "Enrollment is required to receive Medicare benefits."
        ]
    },
    {
        "id": 20,
        "question": "What is Medicaid?",
        "ground_truth": "Medicaid is a government program providing health coverage to low-income individuals and families.",
        "contexts": [
            "Eligibility depends on income and household size.",
            "Medicaid covers doctor visits, hospital care, and long-term care."
        ]
    }
]

with open("evaluation_data.json", "w", encoding="utf-8") as f:
    json.dump(evaluation_data, f, indent=4, ensure_ascii=False)

print("✅ evaluation_data.json with 20 entries created successfully!")
