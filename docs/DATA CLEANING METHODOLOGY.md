# Data Cleaning Methodology Documentation

This document outlines the sophisticated data cleaning and processing methodology used for creating training datasets for the FinSight AI model. Our approach involves three complementary data sources, each with tailored processing pipelines optimized for financial domain-specific training.

## 1. Reddit Financial Data Processing (`prepare_reddit_data.py`)

### Data Source
- Reddit financial discussions and comments from finance-related subreddits
- Raw data collected in JSONL format with comprehensive post metadata
- Focus on high-quality, informative responses with financial relevance

### Cleaning Pipeline

#### Initial Loading & Filtering
- Load JSONL data with error handling for malformed entries
- Score-based filtering using composite metrics:
  - `z_score`: Statistical normalization of comment scores
  - `combined_score`: Weighted scoring incorporating multiple engagement factors
  - `comment_normalized_score`: Context-aware score normalization
- Filter to retain only top 80% of content based on quality metrics

#### Content Cleaning & Formatting
- **Enhanced Text Cleaning**:
  - Remove markdown-style links and plain URLs
  - Clean Reddit-specific formatting (bold, italic, strikethrough)
  - Remove quote blocks and special characters
  - Eliminate Reddit artifacts (edits, updates, TLDRs)
  - **Advanced sentence grouping**: Group every 4 sentences with paragraph breaks for improved readability
  - Ensure proper capitalization and punctuation
  - Normalize whitespace while preserving paragraph structure

#### Quality Assessment
- Multi-dimensional quality scoring:
  - Word length analysis
  - Sentence structure coherence
  - Punctuation and capitalization validation
  - Conversational style assessment
  - Profanity detection (17+ pattern categories)
  - Financial relevance evaluation

#### Advanced Filtering Techniques
- **Text Quality**: Threshold filtering (>0.6) based on composite quality score
- **Complexity Analysis**: Lexical diversity, sentence length, vocabulary sophistication (>0.3)
- **Financial Relevance**: Multi-layer domain relevance scoring:
  - Keyword density analysis (600+ financial terms dictionary)
  - Entity recognition with domain-weighted scoring
  - Semantic similarity to financial topics
  - Combined weighted scoring (>0.2)
- **Conversational Style**: Pattern-based detection of dialogue markers
- **Content Safety**: Multi-pattern profanity and inappropriate content filtering
- **Length Constraints**: 50-2000 character boundaries for optimal training

#### Deduplication Strategy
- TF-IDF vectorization of content
- Cosine similarity calculation with batch processing
- Threshold-based similar content identification (>0.85 similarity)
- Efficient removal of near-duplicate responses

#### Conversation Construction
- Advanced multi-turn conversation generation:
  - 80% with conversational starters, 20% direct Q&A
  - Dynamic greeting and introduction variations
  - Controlled sample usage (max 5 uses per response)
  - Conversation length randomization (3-7 turns)
  - Integration with intro dataset templates for natural flow

## 2. Company Q&A Processing (`prepare_company_qa.py`)

### Data Source
- Financial Q&A dataset derived from 10-K filings
- Company-specific questions and answers with contextual information
- Structured data with ticker symbols and filing year references

### Processing Methodology

#### Data Organization & Enrichment
- Group by company ticker for contextual consistency
- Maintain filing year context for temporal relevance
- Company name mapping for natural language references
- Sample usage tracking to prevent overrepresentation

#### Conversation Types & Generation
- **Multi-turn Conversations**:
  - Professional financial advisor system prompt
  - Contextually relevant greeting sequences
  - Fact-based company question incorporation
  - Dynamic conversation length (3-7 turns)
  - Company name personalization in questions
  
- **Cross-Company Conversations**:
  - Comparative analysis between related companies
  - Structured company comparison templates
  - Aspect-based comparison framework (14+ comparison aspects)
  - Balanced representation of both companies
  
- **List-based Responses**:
  - Specialized list formatting system prompts
  - Multiple list style variations (numbered, bullet, detailed)
  - Template-based list generation with topic context
  - Minimum content quality validation for list items
  - Structured comparison formatting

#### Advanced Feature Generation
- **Content Combination**:
  - Strategic Q&A pair combining for comprehensive responses
  - Four combination templates for natural transitions
  - Context preservation across combined content
  
- **Question Personalization**:
  - Pattern-based company reference replacement
  - Natural language transformation for company references
  - Addition of company context when absent

#### Metadata & Validation
- Unique conversation ID generation using SHA-256 hashing
- Rich metadata inclusion (ticker, company name, filing year)
- Format validation and content quality assurance
- Role-based message structure validation

## 3. Introductory Conversations (`create_intro_dataset.py`)

### Purpose
- Establish AI assistant identity as a financial expert
- Create natural conversation starters with proper tone
- Set expectations about AI capabilities and expertise
- Provide coherent first-turn interactions

### Generation Methodology

#### Template Diversity
- **Greeting Variations**:
  - 70+ diverse greeting patterns spanning formal to casual styles
  - Intentional spelling/capitalization variations for robustness
  - Context-appropriate introductions

- **Response Templates**:
  - Professional, confidence-inspiring responses
  - Consistent AI identity as "FinSight"
  - Finance-focused introductions with area specializations
  - Expertise demonstration through structured capability lists

#### Question-Response Categories
- **Name/Identity Questions**: 
  - Who/what are you questions with consistent responses
  - Brand identity reinforcement in answers
  - Tailored explanation of the name "FinSight"

- **Capability Questions**:
  - What can you do/help with questions
  - Structured capability responses with financial focus
  - Bullet and numbered list formatting variations
  - Expertise area categorization

- **Model Explanation Questions**:
  - AI nature and limitations explanations
  - Training and knowledge description
  - Professional boundaries and expertise scope definition

## Combined Dataset Integration

### Merging Strategy
- **Proportional Source Integration**:
  - Reddit financial discussions (~70% of final dataset)
  - Company Q&A data (~20% of final dataset)
  - Introductory conversations (~10% of final dataset)

- **Quality-Balanced Selection**:
  - Prioritize highest quality examples from each source
  - Ensure domain coverage across financial topics
  - Balance formal and conversational content
  - Maintain consistent AI persona across sources

### Processing Optimizations
- **Performance Enhancements**:
  - Parallel text processing using ProcessPoolExecutor
  - Granular caching of intermediate processing stages
  - Batch processing for memory-intensive operations
  - Progress tracking with rich logging

- **Technical Implementation**:
  - Consistent UTF-8 encoding
  - Standardized JSONL output format
  - Error handling with informative logging
  - Processing pipeline resilience

## Quality Metrics & Validation

### Content Quality Assessment
1. **Text Quality Metrics**:
   - Average word length analysis
   - Sentence structure coherence
   - Punctuation and capitalization validation
   - Minimum word count enforcement
   - Content appropriateness filtering

2. **Financial Relevance Scoring**:
   - Financial keyword density (600+ terms dictionary)
   - Domain-specific entity recognition
   - Semantic similarity to financial topics
   - Combined weighted relevance score

3. **Complexity Analysis**:
   - Lexical diversity measurement
   - Sentence length variability
   - Vocabulary sophistication assessment
   - Content structure evaluation

### Technical Validation
1. **Format Integrity**:
   - JSON structure validation
   - Role alignment verification
   - Message content type checking
   - Metadata completeness verification

2. **Conversation Quality**:
   - Message sequence coherence
   - System prompt consistency
   - Contextual relevance between turns
   - Professional tone maintenance

## Output Format

### Standard Structure
```json
{
    "messages": [
        {
            "role": "system",
            "content": "You are FinSight, an AI financial advisor. Provide accurate and helpful financial guidance."
        },
        {
            "role": "user",
            "content": "User message content"
        },
        {
            "role": "assistant",
            "content": "Assistant response content"
        }
        // Additional turns as needed
    ],
    "metadata": {
        "source": "reddit: subreddit_name | company_qa | intro_dataset",
        "conversation_id": "unique_hash_identifier",
        "additional_fields": "source-specific metadata"
    }
}
```

## Content Structuring Innovations

### Enhanced Readability
- **Paragraph Grouping**: Automatic grouping of every 4 sentences with paragraph breaks
- **List Formatting**: Multiple list styles (numbered, bullet, detailed) for structured information
- **Comparison Structures**: Specialized formats for comparing financial entities

### Response Variation
- Rich mix of direct answers, list-based responses, and multi-part explanations
- Natural transitions between topics in multi-turn conversations
- Professional tone with consistent expertise demonstration

## Future Enhancements

1. **Dataset Expansion Opportunities**
   - Integration of earnings call transcripts for technical financial language
   - Incorporation of financial news analysis for current events context
   - Expansion of cross-asset class discussions (stocks, bonds, crypto, etc.)

2. **Processing Improvements**
   - Enhanced semantic analysis of financial concepts
   - More sophisticated entity relationship mapping
   - Advanced financial sentiment analysis
   - Temporal awareness of market conditions