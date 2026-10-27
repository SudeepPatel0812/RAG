"""
Practice data for a ContextualCompressionRetriever.

Each document is LONG and covers several topics, but a question usually needs
only ONE or TWO sentences of it. A normal retriever returns the whole document
(lots of noise in the prompt). A compression retriever post-processes the results:
    - extractor (LLMChainExtractor) -> keeps only the relevant sentences
    - filter (LLMChainFilter / EmbeddingsFilter) -> drops whole documents that
      aren't relevant, even if their vector was close

Some documents are "near misses": similar words, wrong answer. Those are what the
filter should drop.

Usage (from a script in this folder):
    from compression_data import documents, test_questions
"""

documents = [
    {
        "id": "handbook_001",
        "text": (
            "Welcome to Northwind. Our office opens at 8am and closes at 7pm on weekdays. "
            "Visitors must sign in at reception and wear a badge at all times. "
            "Full-time employees receive 25 days of paid annual leave per year, plus public holidays. "
            "The cafeteria serves lunch between 12pm and 2pm and offers vegetarian options daily. "
            "Parking spaces are allocated by seniority and requested through Facilities."
        ),
        "metadata": {"source": "employee_handbook", "section": "general"},
    },
    {
        "id": "handbook_002",
        "text": (
            "Northwind values continuous learning. Each employee has an annual training budget of $1,500 "
            "for courses, books and conferences. "
            "Performance reviews take place twice a year, in March and September. "
            "Salary increases are decided after the September review and applied from January. "
            "Employees who refer a candidate who is hired receive a $2,000 referral bonus after three months."
        ),
        "metadata": {"source": "employee_handbook", "section": "growth"},
    },
    {
        "id": "it_policy_001",
        "text": (
            "All laptops are encrypted with BitLocker and must be locked when unattended. "
            "Passwords must be changed every 90 days and cannot reuse the last 5 passwords. "
            "Guests can connect to the 'Northwind-Guest' Wi-Fi network; its password changes every Monday "
            "and is shown on the screen at reception. "
            "Personal USB drives are blocked on company devices for security reasons."
        ),
        "metadata": {"source": "it_policy", "section": "security"},
    },
    {
        "id": "it_policy_002",
        "text": (
            "Software requests are made through the Service Desk and approved by your manager. "
            "Standard approved tools include Slack, Zoom, VS Code and the Microsoft Office suite. "
            "Generative AI tools may be used only with non-confidential data, and customer data must never "
            "be pasted into public AI chatbots. "
            "Monitors, keyboards and other peripherals can be ordered from the internal IT shop."
        ),
        "metadata": {"source": "it_policy", "section": "software"},
    },
    {
        "id": "finance_001",
        "text": (
            "The financial year runs from April to March. "
            "Purchase orders above $5,000 require approval from the department head and Finance. "
            "Corporate credit cards are issued to managers and frequent travellers only. "
            "Invoices from suppliers are paid within 45 days of receipt. "
            "Business travel meals are covered by a per diem of $75 per day for domestic trips."
        ),
        "metadata": {"source": "finance_policy", "section": "general"},
    },
    {
        "id": "safety_001",
        "text": (
            "Fire drills are held every quarter; follow the green exit signs to the assembly point in the car park. "
            "First aid kits are located next to every kitchen. "
            "Report any workplace injury to your manager and to HR within 24 hours. "
            "The building is accessible, with lifts and ramps at every entrance. "
            "Smoking is allowed only in the designated area behind building B."
        ),
        "metadata": {"source": "safety_guide", "section": "general"},
    },
    # ---------- near misses: similar words, wrong answer ----------
    {
        "id": "customer_001",
        "text": (
            "Northwind customers on the Pro plan get 25 days of free premium support per year. "
            "Customer passwords can be reset from the login page using the 'Forgot password' link. "
            "Our customer Wi-Fi hotspots are available in all partner stores."
        ),
        "metadata": {"source": "customer_faq", "section": "support"},
    },
    {
        "id": "blog_001",
        "text": (
            "In this blog post our CTO shares how the team learned to train machine learning models. "
            "The training took three months and used a budget of $50,000 in cloud credits. "
            "The new model improved recommendation accuracy by 12%."
        ),
        "metadata": {"source": "company_blog", "section": "engineering"},
    },
]

# Each question is answered by ONE sentence inside a long document.
# expected_id = the document that holds it, answer = what a good extractor keeps.
test_questions = [
    {
        "question": "How many vacation days do I get?",
        "expected_id": "handbook_001",
        "answer": "25 days of paid annual leave per year",
        "trap": "customer_001 also says '25 days' (customer support, not leave)",
    },
    {
        "question": "What is the guest Wi-Fi password?",
        "expected_id": "it_policy_001",
        "answer": "changes every Monday, shown at reception",
        "trap": "customer_001 mentions customer Wi-Fi hotspots",
    },
    {
        "question": "How much money can I spend on training courses?",
        "expected_id": "handbook_002",
        "answer": "$1,500 annual training budget",
        "trap": "blog_001 talks about 'training' ML models with a $50,000 budget",
    },
    {
        "question": "Can I use ChatGPT at work?",
        "expected_id": "it_policy_002",
        "answer": "only with non-confidential data, never customer data",
        "trap": "no document says 'ChatGPT', the extractor must understand 'generative AI tools'",
    },
    {
        "question": "When do I get a pay rise?",
        "expected_id": "handbook_002",
        "answer": "decided after the September review, applied from January",
        "trap": "different words: 'pay rise' vs 'salary increases'",
    },
]
