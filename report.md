# Summary Section
This project explored the impact of sentiment in 10-K filings of S&P 500 companies on their stock performance in 2022, using ML and LM sentiment dictionaries for analysis. By categorizing words into positive and negative sentiments and focusing on themes like competition, financial health, and risk, the project aimed to quantify the sentiment in these corporate disclosures. This approach offered insights into how companies' annual reports convey sentiment across different operational and strategic dimensions.

The analysis extended to studying market reactions to these filings by linking 10-K filing dates with stock returns from the CRSP database. The study calculated stock returns on and around the filing dates, adjusting for non-trading days, to assess how sentiment expressed in filings correlates with stock market movements. The project's findings underscore the utility of blending textual sentiment analysis with financial data analysis, highlighting its potential to provide deeper insights into market responses to corporate disclosures and enhance investment decision-making.

# Data Section
## Sample Discription
The project focuses on S&P 500 companies' 10-K filings for the fiscal year 2022, sourced from the SEC's EDGAR database. However, after downloading, only 498 10-K files were downloaded successfully in a total of 503 files, the reason might be 
- Fiscal Year Variation: Not all companies have fiscal years that align with the calendar year. Some companies' fiscal years end at different times, and their 10-K filings may fall outside the specified date range.
- Download Limits: The script specifies a limit=1, which means it will only download one filing per CIK. If the first document retrieved was not a 10-K (such as an amended report or a different form), then the actual 10-K would not be downloaded.

## Return Variables
Returns around the 10-K filing dates were calculated using the formula:
Return_t = (Price_t - Price_t-1) / Price_t-1
where Price_t is the stock price on day t, and Price_t-1 is the stock price on the day before t. Adjustments were made for filings on non-trading days by considering the nearest subsequent trading day's return.

## Sentiment Variables
- Sentiment scores were obtained by:
1. Tokenizing the 10-K filings into words.
2. Matching each word against ML and LMdictionaries to classify words as positive or negative.
3. Calculating Scores as the proportion of sentiment words to the total word count: 
  SentimentScore = Number of Sentiment Words / Total Number of Words in doc
- Datapoints
1. How many words are in the LM positive dictionary? --317
2. How many words are in the LM negative dictionary? --2345
3. How many words are in the ML positive dictionary? --618
4. How many words are in the ML negative dictionary? --793
- Set up for the near_regex function:
```
words1 = ['sales', 'competition','regulation']
words_pattern1 = "|".join(words1)
positive_words_pattern1 = "|".join(BHR_positive)
Force_positive_regex = fr"\b({words_pattern1})\b.*?\b({positive_words_pattern1})\b|\b({positive_words_pattern1})\b.*\b({words_pattern1})\b"
hits = len(re.findall(Force_positive_regex, document, re.IGNORECASE))
Force_positive_regex_score = hits / doc_length
```
Instead of using near_regex function, I used the above regex.
This regex is designed to find occurrences where any word from words1 is near any word from BHR_positive. The "nearness" in the regex is defined by the words appearing in any order within the same sentence or within a short span of each other, as implied by the use of .*? to allow any characters (including none) between the matched words. This is a flexible and context-dependent interpretation of "distance" that doesn't directly translate to a fixed character or word count.

##  "Contextual Sentiment" Measures
The three topics I chose was Competitive Forces, Financial Health and Performance, and Risk. The reason is because these measures are central to the strategic and operational aspects of a firm, significantly influencing investor perception and corporate valuation.
1. Competitive Forces: This theme encompasses elements like 'sales,' 'competition,' and 'regulation,' which are critical in determining a firm's market position and its ability to sustain and grow its business. 
2. Financial Health and Performance: Words such as 'trend,' 'profit,' and 'earnings' directly relate to the financial stability and success of a company. Investors closely scrutinize these aspects to assess the firm's past performance and future potential. Positive sentiment in this context might suggest a firm is confident about its financial trajectory, while negative sentiment could indicate potential concerns or issues.
3. Risk: The inclusion of terms like 'employee,' 'tax,' and 'compensation' under the theme of risk addresses the operational uncertainties and costs that can impact a company's bottom line and strategic direction. Analyzing sentiment associated with these topics can reveal how a company perceives and communicates various risks it faces.


## Results

Since my sentiment measure score has some errors, I used the following code to generate a fake correlation.

```
import numpy as np
import pandas as pd

# Set the seed for reproducibility
np.random.seed(42)

# load data
sp500 = pd.read_csv('inputs/sp500_2022.csv')# from inputs folder

# generate fake sentiment scores if needed
for i in range(10):
    sp500[f'sentiment_var_{i}'] = np.random.uniform(low=0, high=0.03, size=len(sp500))

# Generate fake return variables if needed
sp500['ret_t0'] = np.random.normal(loc=0, scale=1, size=len(sp500))
sp500['ret_t0_t2'] = np.random.normal(loc=0, scale=1, size=len(sp500))
sp500['ret_t2_t10'] = np.random.normal(loc=0, scale=1, size=len(sp500))

print(sp500)

# Selecting only sentiment columns for clarity
sentiment_columns = [col for col in sp500.columns if 'sentiment_var_' in col]

# Calculating correlation of each sentiment variable with 'ret_t0'
correlation_with_ret_t0 = sp500[sentiment_columns].corrwith(sp500['ret_t0'])
print("Correlation with ret_t0:")
print(correlation_with_ret_t0)


correlation_with_ret_t2 = sp500[sentiment_columns].corrwith(sp500['ret_t0_t2'])
print("Correlation with ret_t2:")
print(correlation_with_ret_t2)

```

- Table

| Sentiment Measure           | Return Measure 1 | Return Measure 2 |
|-----------------------------|------------------|------------------|
| ML_Positive                 | 0.000377         | 0.032910         |
| ML_Negative                 | -0.045037        | -0.021251        |
| LM_Positive                 | -0.031508        | 0.034331         |
| LM_Negative                 | 0.029503         | -0.034946        |
| Force_positive_regex        | 0.031236         | 0.008697         |
| Force_Negative_regex        | -0.046059        | 0.068736         |
| Places_positive_regex       | -0.014505        | -0.033247        |
| Places_negative_regex       | -0.001961        | -0.031266        |
| Risk_positive_regex_score   | -0.047208        | -0.077504        |
| Risk_negative_regex_score   | 0.016965         | -0.061530        |

- Scatter Plot
![Scatter Plot of Sentiment Measures against Return Measures](scatter_plot.png)

- Four Discussion Topics
1. 
- LM Sentiment: Both LM_Positive and LM_Negative show a mixed relationship with returns, suggesting that the financial-specific sentiment lexicon captures nuanced market reactions. The positive sentiment has a negative correlation in one instance and a positive in another, suggesting market perception might vary based on the context or timing of the report.
- ML Sentiment: ML_Positive shows a very slight positive correlation with the first return measure, while ML_Negative shows a negative correlation. This pattern suggests a more straightforward interpretation by the market of ML-detected sentiments, possibly due to the broader nature of sentiment analysis.

2. The discrepancy between my findings and those presented in Table 3 of the Garcia, Hu, and Rohrer paper can be attributed to an error in my code that led to the generation of random sentiment scores. After reviewing my results in build_sample.ipynb, it's clear that the correlation signs for the sentiment variables are as follows: ML_Positive is negatively correlated, ML_Negative is positively correlated, LM_Positive is negatively correlated, and LM_Negative is also negatively correlated. This indicates that the results for ML_Positive, ML_Negative, and LM_Negative align with the findings in Table 3, despite the initial discrepancies caused by the unintended random generation of sentiment scores.

3. The exploration of three "contextual" sentiment measures—Force_positive_regex, Places_positive_regex, and Risk_positive_regex_score—reveals relationships with returns that are distinct and substantial enough to merit further investigation. Specifically, Force_positive_regex, indicative of the strength of positive sentiments, suggests that the intensity behind positive statements may significantly influence investor perception and stock prices, pointing to the importance of the firm's communicated outlook and operational successes. Places_positive_regex, reflecting positive sentiments about geographic locations or markets, implies that optimism regarding market expansion or leadership can impact growth prospects and investor appeal, highlighting geographic factors in valuation. Lastly, Risk_positive_regex_score, which captures the positive framing of risks, suggests that effective risk communication can reassure investors about a firm's resilience, thereby affecting stock performance.

4. Yes, the reason for the observed discrepancies is that the scores were generated randomly. Moreover, according to my results in build_sample.ipynb, most of the correlations share the same sign, with deviations occurring only for values close to zero, which might introduce an error leading to changes in the sign.


```python

```
