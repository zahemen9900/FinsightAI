# Data Cleaning Methodology Documentation

This document outlines the data cleaning and processing methodology used for creating training datasets for the FinSight AI model. The process involves three main data sources, each with its own cleaning pipeline.

## 1. Reddit Financial Data Processing (`prepare_reddit_data.py`)

### Data Source
- Reddit financial discussions and comments
- Focus on finance-related subreddits
- Raw data collected through Reddit API

### Cleaning Pipeline
1. **Initial Filtering**
   - Remove deleted/removed comments
   - Filter out comments shorter than minimum length
   - Remove comments with excessive special characters

2. **Content Cleaning**
   - Remove URLs and links
   - Clean special characters and emojis
   - Normalize whitespace and newlines
   - Remove markdown formatting

3. **Quality Checks**
   - Language detection (ensure English content)
   - Sentiment analysis to filter extreme content
   - Remove duplicate content
   - Filter out bot responses and automated messages

4. **Conversation Structure**
   - Create conversation pairs from comment chains
   - Ensure context preservation
   - Maintain thread hierarchy information
   - Add metadata for tracking

## 2. Company Q&A Processing (`prepare_company_qa.py`)

### Data Source
- Financial Q&A dataset based on 10-K filings
- Structured question-answer pairs about company financials

### Processing Methodology
1. **Data Organization**
   - Group by company ticker
   - Maintain filing year information
   - Track context relationships

2. **Conversation Generation**
   - Create multi-turn conversations (3-7 turns)
   - Generate cross-company conversations
   - Combine related Q&A pairs
   - Include greetings and conversation starters

3. **Context Management**
   - Strategic context inclusion (80% probability)
   - Context mixing for cross-company conversations
   - Maintain context relevance

4. **Quality Control**
   - Validate conversation formats
   - Clean message content
   - Generate unique conversation IDs
   - Track sample usage to prevent overuse

5. **Output Structure**
   - JSONL format with metadata
   - System prompts customization
   - Role-based message structure
   - Comprehensive validation checks

## 3. Introductory Conversations (`create_intro_dataset.py`)

### Purpose
- Create natural conversation starters
- Establish AI assistant identity
- Set appropriate conversation tone

### Processing Steps
1. **Template Creation**
   - Diverse greeting patterns
   - Professional responses
   - Consistent AI identity
   - Finance-focused introductions

2. **Variation Generation**
   - Multiple conversation paths
   - Different greeting styles
   - Dynamic response patterns
   - Context-aware transitions

3. **Quality Assurance**
   - Consistency checks
   - Professional tone maintenance
   - Clear role establishment
   - Format validation

## Combined Dataset Integration

### Merging Strategy
1. **Proportional Mixing**
   - 67% Reddit financial discussions (_~10k samples_)
   - 100% Company Q&A data (_~5k samples_)
   - 100% Introductory conversations(_~5k samples_)

2. **Quality Balancing**
   - Maintain conversation naturalness
   - Ensure professional tone
   - Preserve domain expertise
   - Balance formal and informal content

3. **Format Standardization**
   - Consistent JSON structure
   - Unified metadata schema
   - Standard role definitions
   - Common validation criteria

## Validation and Quality Metrics

### Dataset Quality Checks
1. **Content Validation**
   - Message format consistency
   - Role alignment
   - Context relevance
   - Professional tone

2. **Technical Validation**
   - JSON structure integrity
   - Character encoding
   - Metadata completeness
   - ID uniqueness

3. **Domain-Specific Checks**
   - Financial terminology accuracy
   - Context appropriateness
   - Professional language
   - Information accuracy

## Output Format

### Standard Structure
```json
{
    "messages": [
        {
            "role": "system/user/assistant",
            "content": "message text"
        }
    ],
    "metadata": {
        "source": "source_identifier",
        "conversation_id": "unique_id",
        "type": "conversation_type",
        "additional_metadata": "..."
    }
}
```

## Best Practices and Guidelines

1. **Content Quality**
   - Maintain professional tone
   - Ensure financial accuracy
   - Preserve context relevance
   - Balance formality levels

2. **Technical Standards**
   - Consistent encoding (UTF-8)
   - Proper JSON formatting
   - Robust error handling
   - Comprehensive logging

3. **Data Privacy**
   - Remove personal identifiers
   - Sanitize sensitive information
   - Maintain data anonymity
   - Follow data protection guidelines

## Future Improvements

1. **Potential Enhancements**
   - Expanded cross-company interactions
   - More diverse conversation patterns
   - Enhanced context integration
   - Improved validation metrics

2. **Monitoring and Updates**
   - Regular quality assessments
   - Dataset expansion opportunities
   - Validation criteria updates
   - Processing pipeline optimization
