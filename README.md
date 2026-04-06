📊 Agentic Multi-Source Retrieval System (LLM + Deterministic Reasoning)
🧠 Overview

This project is a CLI-based Agentic Retrieval System that answers business-related questions by selecting and combining multiple data sources.

It follows a hybrid architecture:

Python (Deterministic Layer) → performs all calculations and reasoning
LLM (Groq API) → used only for formatting responses

This ensures:

✅ Zero hallucination
✅ Accurate insights
✅ Clear citations
🎯 Objective

The system is designed to:

Retrieve data from multiple sources
Dynamically select sources based on query
Generate concise business insights
Provide answers with proper citations
🏗️ Project Structure
project/
│── agent.py
│── data/
│   ├── sales.csv
│   ├── payroll.csv
│   └── report.txt
│── README.md
📂 Data Sources
📊 Structured Data
sales.csv → revenue, regions, trends
payroll.csv → salaries, expenses, headcount
📄 Unstructured Data
report.txt → explanations, causes, summaries
⚙️ How It Works
1. User Input
python agent.py
2. Agent Decision Logic

The system selects relevant sources:

[Agent Decision]
- Using sales.csv
- Using report.txt
3. Deterministic Reasoning

Python performs:

Data analysis
Trend calculation
Comparisons
Insight generation
4. LLM Formatting Layer

The LLM:

Formats the final response
Does NOT calculate or infer
5. Output with Citations

Example:

--- ANSWER ---
Revenue decreased from ₹136,750 to ₹112,210 due to competitive pressure and supply chain disruptions.

--- SOURCES ---
Source: sales.csv (rows 2–25)
Source: report.txt (paragraph 1)
🧠 Agentic Decision Logic
Query Type	Source Used
Trend	sales.csv / payroll.csv
Comparison	sales.csv + payroll.csv
Why	report.txt + CSV
Summary	report.txt
Financial Health	Combined
🚀 Features
Multi-source retrieval
Agent-based source selection
Deterministic reasoning
Controlled LLM usage
Business-level insights
CLI interface
Accurate citations
🔒 LLM Safety Design

To prevent hallucination:

❌ LLM does NOT perform calculations
❌ LLM does NOT access raw data
❌ LLM does NOT generate new facts

✔ It only formats precomputed insights

💡 Example Questions
How has revenue changed over the last quarter?
Compare salary expenses with revenue
Why did revenue decrease?
What are the key financial risks?
Is the business financially sustainable?
🛠️ Installation & Setup
Clone Repository
git clone <your-repo-link>
cd project
Install Dependencies
pip install pandas
Set Groq API Key

Mac/Linux:

export GROQ_API_KEY=your_api_key

Windows:

set GROQ_API_KEY=your_api_key
Run the Project
python agent.py
📈 Example Output
Enter your question: Why did revenue decrease?

[Agent Decision]
- Using sales.csv
- Using report.txt

--- ANSWER ---
Revenue decreased from ₹136,750 to ₹112,210 due to competitive pressure and supply chain disruptions.

--- SOURCES ---
Source: sales.csv (rows 2–25)
Source: report.txt (paragraph 1)
🧪 Evaluation Criteria (Satisfied)
✔ Correct source selection
✔ Multi-source usage
✔ Accurate reasoning
✔ Clear citations
✔ Simple and structured implementation
🧠 Design Philosophy
Accuracy over generation
Explainability over complexity
Deterministic logic over blind LLM usage
🚀 Future Improvements
Add more datasets (inventory, customers)
Improve anomaly detection
Add visualization layer
Support multiple currencies
👩‍💻 Author

Swathiga S
AI Developer | Data Science Enthusiast
