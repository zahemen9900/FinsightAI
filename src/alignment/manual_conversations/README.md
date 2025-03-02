
# Manual Finance Conversation Generation

This directory contains tools for generating finance-related questions that can be manually answered to create high-quality training data for financial AI assistants.

## Key Features

- Generate diverse, realistic finance questions across multiple categories
- Output in a simple format ready for manual completion
- Customizable question counts and categories
- No auto-generated responses (to be filled in manually)

## Usage

### Generate Finance Questions

Generate a set of finance questions without answers:

```bash
python generate_finance_questions.py --output /home/zahemen/datasets/finance_questions.txt --count 100
```

This will create a text file with questions formatted as:

```
User: How do Apple and Microsoft compare in terms of financial performance?
Assistant: [leave-blank]

User: What's the difference between ETFs and mutual funds?
Assistant: [leave-blank]

...
```

### Question Categories

The generator creates questions in these categories:

1. Company-specific questions
2. Company comparison questions
3. Investment strategy questions
4. Financial planning questions
5. Market analysis questions
6. Financial product questions
7. Advanced financial concept questions

### Creating High-Quality Answers

After generating the questions, you can:

1. Fill in the `[leave-blank]` sections with expert financial answers
2. Add follow-up questions and answers to create multi-turn conversations
3. Incorporate real-world data and specific examples in answers
4. Ensure responses are accurate, balanced, and informative

## Adding to Training Data

Once completed, these manually crafted conversations can be processed into the proper format for model fine-tuning.
