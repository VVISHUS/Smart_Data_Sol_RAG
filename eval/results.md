# Evaluation Results

**19/20 passed (95%)**

| category | passed | total | accuracy |
|---|---|---|---|
| figure | 1 | 1 | 100% |
| refusal | 2 | 2 | 100% |
| table | 10 | 10 | 100% |
| table_multi | 3 | 3 | 100% |
| text | 3 | 4 | 75% |

## Per-question detail

### q01 (table) - PASS

**Q:** What were Apple's total net sales for the three months ended June 25, 2022?

**A:** Apple's total net sales for the three-month period ended June 25, 2022 (found in the current-period column for the three months ended June 25, 2022) were **$82,959 million** [table on p.18 (table 1)].

*retrieved:* text=4, table=4 · *pages:* 3, 4, 5, 6, 10, 18 · *verdict:* ok

### q02 (table) - PASS

**Q:** What were iPhone net sales in the third quarter of fiscal 2022?

**A:** Assuming the three-month (quarterly) period ended June 25, 2022, iPhone net sales were $40,665 million in the current year column [table on p.10 (table 1)].

*retrieved:* text=5, table=4 · *pages:* 4, 5, 6, 10, 18, 19, 20 · *verdict:* ok

### q03 (table) - PASS

**Q:** How much did Apple spend on research and development in the quarter ended June 25, 2022?

**A:** Based on the three-month period ended June 25, 2022 (current year column), Apple spent $6,797 million on research and development [text on p. 21 (narrative)].

*retrieved:* text=6, table=4 · *pages:* 3, 4, 5, 6, 8, 10, 18, 21 · *verdict:* ok

### q04 (table) - PASS

**Q:** What was Apple's net income for the three months ended June 25, 2022?

**A:** For the three months ended June 25, 2022 (the current-year three-month period), Apple's net income was $19,442 million [text on p.5].

*retrieved:* text=6, table=4 · *pages:* 3, 4, 5, 6, 7, 9, 10, 18, 21 · *verdict:* ok

### q05 (table) - PASS

**Q:** What were Apple's total assets as of June 25, 2022?

**A:** As the question does not specify a period, we assume the balance sheet figure as of June 25, 2022. Apple's total assets as of June 25, 2022, were $336,309 million [table on p.6 (table 1)].

*retrieved:* table=4, text=5 · *pages:* 3, 5, 6, 7, 8, 10, 18 · *verdict:* ok

### q06 (table) - PASS

**Q:** How much cash and cash equivalents did Apple hold as of June 25, 2022?

**A:** As of June 25, 2022, Apple held $27,502 million in cash and cash equivalents, as reported in the current column of the June 25, 2022 balance sheet [table on p.6 (table 1)]. Additionally, Note 3 reports that cash and cash equivalents included $12,852 million of "Cash" as of June 25, 2022 [text on p.11 (narrative)].

*retrieved:* text=5, table=4 · *pages:* 3, 5, 6, 8, 10, 11, 18 · *verdict:* ok

### q07 (table) - PASS

**Q:** What was Apple's diluted earnings per share for the third quarter of 2022?

**A:** Apple's diluted earnings per share for the three-month period ended June 25, 2022 (the third quarter of 2022) was $1.20 [table on p.9 (table 1)].

*retrieved:* text=6, table=4 · *pages:* 4, 5, 6, 7, 9, 10, 17, 18, 21 · *verdict:* ok

### q08 (table) - PASS

**Q:** What were Services net sales in the three months ended June 25, 2022?

**A:** For the three-month period ended June 25, 2022 (assuming the three-month period as no period was specified), Services net sales were $19,604 million [table on p.18 (table 1)].

*retrieved:* table=5, text=1 · *pages:* 4, 10, 18, 19, 20 · *verdict:* ok

### q09 (table) - PASS

**Q:** What was Apple's total gross margin percentage for the third quarter of 2022?

**A:** For the three-month period (quarter) ended June 25, 2022, Apple's total gross margin percentage was 43.3% (reported in the current-year three-month column) [table on p.20 (table 2)].

*retrieved:* text=6, table=4 · *pages:* 4, 5, 10, 16, 17, 18, 20, 21 · *verdict:* ok

### q10 (table) - PASS

**Q:** What were net sales in Greater China for the three months ended June 25, 2022?

**A:** Assuming the three-month (quarterly) period, net sales in Greater China for the three months ended June 25, 2022 were $14,604 million [table on p.19 (table 1)].

*retrieved:* table=6 · *pages:* 4, 5, 16, 18, 19, 21 · *verdict:* ok

### q11 (table_multi) - PASS

**Q:** By what percentage did Mac net sales change in the third quarter of 2022 compared to the same quarter in 2021?

**A:** Assuming the three-month (quarterly) period, Mac net sales decreased by **(10)%** in the third quarter of 2022 compared to the same quarter in 2021 [table on p.18 (table 1)].

*retrieved:* table=4, text=4 · *pages:* 10, 16, 17, 18, 19, 20 · *verdict:* ok

### q12 (table_multi) - PASS

**Q:** What was Apple's effective tax rate in the third quarter of 2022, and how did it compare to the same quarter of 2021?

**A:** Assuming the three-month (quarterly) period:

* In the third quarter of 2022 (Three Months Ended June 25, 2022), Apple's effective tax rate was 15.7% [table on p.21 (table 3)].
* In the same quarter of the prior year (Three Months Ended June 26, 2021), Apple's effective tax rate was 10.8% [table on p.21 (table 3)]. 
* The effective tax rate for the third quarter of 2022 was higher compared to the third quarter of 2021 due primarily to a higher effective tax rate on foreign earnings and an adjustment to prior years' foreign-derived intangible income deductions [text on p.22 (narrative)].

*retrieved:* text=6, table=4 · *pages:* 4, 5, 6, 10, 17, 18, 20, 21, 22 · *verdict:* ok

### q13 (table_multi) - PASS

**Q:** Compare Americas and Europe net sales for the three months ended June 25, 2022.

**A:** Assuming the three-month period (quarterly) figures, for the three months ended June 25, 2022, the net sales were:

* **Americas:** $37,472 million (current year column of the three-month period) [table on p.19 (table 1)]
* **Europe:** $19,287 million (current year column of the three-month period) [table on p.19 (table 1)]

Comparing the two, Americas net sales exceeded Europe net sales by $18,185 million [calculated from $37,472 million minus $19,287 million].

*retrieved:* table=6 · *pages:* 4, 16, 18, 19, 20, 21 · *verdict:* ok

### q14 (text) - PASS

**Q:** What did Apple say about the impact of the COVID-19 pandemic on its suppliers?

**A:** Based on the provided filing, Apple stated that certain of its outsourcing partners, component suppliers, and logistical service providers experienced disruptions during the COVID-19 pandemic, resulting in supply shortages, and that similar disruptions could occur in the future [text on p.18 (narrative)].

*retrieved:* text=6, table=4 · *pages:* 3, 5, 6, 10, 17, 18, 22 · *verdict:* ok

### q15 (text) - PASS

**Q:** What is the status of the Epic Games lawsuit against Apple?

**A:** Based on the provided document, the status of the Epic Games lawsuit is as follows:

* Epic Games filed a lawsuit against Apple in the U.S. District Court for the Northern District of California (the "Northern California District Court") alleging violations of federal and state antitrust laws and California's unfair competition law regarding the operation of the App Store [text on p.23 (narrative)]. Apple filed a counterclaim for breach of contract [text on p.23 (narrative)]. 
* On September 10, 2021, the Northern California District Court ruled in favor of Apple on nine out of the ten counts in Epic's claim, and in favor of Apple on its breach of contract claims [text on p.23 (narrative)]. 
* The court found that certain provisions of Apple’s App Store Review Guidelines violate California’s unfair competition law and issued an injunction [text on p.23 (narrative)]. 
* Epic appealed the decision, and Apple filed a cross-appeal and has been granted a stay pending the appeal [text on p.23 (narrative)].

*retrieved:* text=6, table=4 · *pages:* 3, 5, 6, 10, 18, 21, 23 · *verdict:* ok

### q16 (text) - FAIL

**Q:** Why did Japan's net sales decrease in the third quarter of 2022?

**A:** Japan's net sales decreased during the third quarter of 2022 compared to the same period in 2021 due primarily to lower net sales of iPhone and iPad [text on p. 19 (narrative)].

*retrieved:* table=4, text=4 · *pages:* 16, 18, 19, 20, 21 · *verdict:* missing ['yen']

### q17 (text) - PASS

**Q:** What new products did Apple introduce at the end of the third quarter of 2022?

**A:** At the end of the third quarter of 2022, Apple introduced an all-new MacBook Air and an updated 13-inch MacBook Pro, both powered by the new Apple M2 chip [text on p.17 (narrative)].

*retrieved:* text=6, table=4 · *pages:* 3, 5, 6, 10, 17, 18, 19, 21 · *verdict:* ok

### q18 (figure) - PASS

**Q:** What charts or figures are included in this document?

**A:** Based on the provided document excerpts, the following figure is included:

* The Apple corporate logo (described as a solid black apple silhouette with a bite taken out of the right side and a separate leaf positioned above it) [figure on p.1 (embedded image 5)].

*retrieved:* table=5, text=1, figure=1 · *pages:* 1, 2, 3, 16, 20, 23 · *verdict:* ok

### q19 (refusal) - PASS

**Q:** How many iPhone units did Apple sell in the third quarter of 2022?

**A:** The provided document excerpts do not contain this information.

*retrieved:* text=6, table=4 · *pages:* 3, 5, 6, 10, 17, 18, 19, 20 · *verdict:* ok

### q20 (refusal) - PASS

**Q:** What is Apple's forecast for iPhone revenue in fiscal 2023?

**A:** The provided document excerpts do not contain this information.

*retrieved:* text=6, table=4 · *pages:* 3, 5, 6, 10, 17, 18, 19, 20 · *verdict:* ok
