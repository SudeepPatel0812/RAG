"""
Practice data for a MultiQueryRetriever.

Each topic's facts are split across several documents that use DIFFERENT words
(reimbursement / per diem / out-of-pocket; laptop / device / workstation), so a
single query often finds only part of the answer. A multi-query retriever asks
the LLM for several rephrasings, searches with each, and merges the results.

Topics: travel & expenses, broken laptop, working from home, leave, plus
distractors that share words but not meaning.

Usage (from a script in this folder):
    from multi_query_data import documents, test_questions
"""

documents = [
    # ---------- travel & expenses ----------
    {
        "id": "exp_001",
        "text": "Out-of-pocket costs incurred while on company business are reimbursed through the ExpenseHub portal. Submit a claim within 30 days of the trip's end date; late claims require director approval.",
        "metadata": {"category": "Finance", "topic": "expenses"},
    },
    {
        "id": "exp_002",
        "text": "Every claim must include an itemised receipt for any purchase above $25. Credit card statements alone are not accepted as proof of payment.",
        "metadata": {"category": "Finance", "topic": "expenses"},
    },
    {
        "id": "exp_003",
        "text": "Meals during business travel are covered by a daily per diem of $75 for domestic trips and $110 for international trips. No receipts are needed for per diem amounts.",
        "metadata": {"category": "Finance", "topic": "travel"},
    },
    {
        "id": "exp_004",
        "text": "Approved claims are paid with the next monthly payroll run, appearing on your payslip as a separate non-taxable line item.",
        "metadata": {"category": "Finance", "topic": "expenses"},
    },
    {
        "id": "exp_005",
        "text": "Flights must be booked through the corporate travel agency at least 14 days in advance. Economy class is standard; business class is allowed only for flights longer than 8 hours.",
        "metadata": {"category": "Finance", "topic": "travel"},
    },
    # ---------- broken laptop ----------
    {
        "id": "it_001",
        "text": "If your company device stops working, raise a P2 ticket on the Service Desk. Hardware faults are diagnosed within one business day.",
        "metadata": {"category": "IT Support", "topic": "hardware"},
    },
    {
        "id": "it_002",
        "text": "While your workstation is being repaired, collect a loaner machine from the IT counter on floor 3. Loaners are pre-configured with VPN and standard software.",
        "metadata": {"category": "IT Support", "topic": "hardware"},
    },
    {
        "id": "it_003",
        "text": "Files saved in your OneDrive folder are synced automatically and can be restored on any machine. Files stored only on the local disk may be lost during a hardware swap.",
        "metadata": {"category": "IT Support", "topic": "backup"},
    },
    {
        "id": "it_004",
        "text": "Physical damage caused by drops or liquid spills is charged to your cost centre at 50% of the repair price. Manufacturing defects are covered at no cost.",
        "metadata": {"category": "IT Support", "topic": "hardware"},
    },
    # ---------- working from home ----------
    {
        "id": "hr_001",
        "text": "Employees may work remotely up to three days per week after completing their probation period. Remote days must be logged in the HR calendar.",
        "metadata": {"category": "HR", "topic": "remote work"},
    },
    {
        "id": "hr_002",
        "text": "Staff based outside the office receive a monthly internet allowance of $40, paid with salary. Home electricity is not reimbursed.",
        "metadata": {"category": "HR", "topic": "remote work"},
    },
    {
        "id": "hr_003",
        "text": "Working from another country for more than 10 days requires approval from HR and Legal because of tax and visa rules.",
        "metadata": {"category": "HR", "topic": "remote work"},
    },
    # ---------- leave ----------
    {
        "id": "hr_004",
        "text": "Full-time staff are entitled to 25 days of annual leave per calendar year. Up to 5 unused days can be carried over to the next year.",
        "metadata": {"category": "HR", "topic": "leave"},
    },
    {
        "id": "hr_005",
        "text": "If you are unwell, notify your manager before 10am. A doctor's note is required for absences longer than 3 consecutive days.",
        "metadata": {"category": "HR", "topic": "leave"},
    },
    # ---------- distractors: similar words, different meaning ----------
    {
        "id": "dis_001",
        "text": "Customers can request a refund for a subscription within 14 days of purchase by contacting Billing Support.",
        "metadata": {"category": "Customer Support", "topic": "refunds"},
    },
    {
        "id": "dis_002",
        "text": "The office kitchen coffee machine is serviced every Monday. Report a broken machine to Facilities, not IT.",
        "metadata": {"category": "Facilities", "topic": "office"},
    },
]

# Questions where one search usually misses part of the answer.
# expected = ids that together answer the question.
test_questions = [
    {
        "question": "How do I get my money back after a work trip?",
        "expected": ["exp_001", "exp_002", "exp_003", "exp_004"],
    },
    {
        "question": "My laptop broke, what should I do?",
        "expected": ["it_001", "it_002", "it_003", "it_004"],
    },
    {
        "question": "What are the rules for working from home?",
        "expected": ["hr_001", "hr_002", "hr_003"],
    },
    {
        "question": "I'm sick, can I take time off?",
        "expected": ["hr_005", "hr_004"],
    },
]
