##                                          Optimal Facility Location 

Some time ago, we went shopping. After discussing with a furniture and decor shop owner their reason for closing their shop and putting their 
remaining inventory on sale, we found that the shop owner believed they'd not picked the best or most optimal location for their shop. 
Sales  had been rather slow; so they were closing the shop and moving it to a location that they had concluded would be 
more ideal for the furniture shop, given the price range of the items and the shoppers' preferences.

What if the shop owner had made the correct analysis and placed their shop in the more optimal location? 
That would have saved them money and time. 

This inspired a project that would determine the optimal location for a shop, given all the possible variables that go
into such determination: income, demographics, distance to shoppers and clientele, etc. Expectedly, small shop owners use rule
of thumb and intuition, and experience and some sales data to determine where to locate their next shop. And giant multi-national shops have analysts using data 
and algorithms to help inform and determine shop locations.

This open-source (MIT License) project intends to replicate the data analysis and algorithmic processes that 
help inform shop locations. For starters, determining shop location demands some economics and econometrics, with some help from
machine learning. We look forward to sharing the data and code as we research and implement the application. 


# References andLiterature Review

Quite a bit of research has been conducted on optimal shop location. We would expect most big retailers to have even more 
detailed and comprehensive models and data sets for their research and decision making. Here, we list some of the research 
to get us started.

1. Optimal Retail Location: Empirical Methodology and Application to Practice: https://bakerretail.wharton.upenn.edu/wp-content/uploads/2017/01/Chloe-Kim_Optimizing-Retail-Locations-Using-an-Empirical-Model_SSRN.pdf
2. DeepStore: An Interaction-Aware Wide&Deep Model for Store Site Recommendation With Attentional Spatial Embeddings: https://hal.science/hal-02321010/file/8-DeepStore.pdf
3. Store Location Selection via Mining Search Query Logs of Baidu Maps: https://arxiv.org/pdf/1606.03662
4. MetaStore: a task-adaptative meta-learning model for optimal store placement with multi-city knowledge transfer: https://hal.science/hal-03363389/file/3447271.pdf
5. Applied Retail Location Models Using Spatial Interaction Tools: https://www.researchgate.net/profile/Morton-Okelly-2/publication/263389703_Applied_retail_location_models_using_spatial_interaction_tools/links/5413ac410cf2bb7347db266c/Applied-retail-location-models-using-spatial-interaction-tools.pdf
6. Location and Agglomeration: The Distribution of Retail and Food Businesses in Dense Urban Environments: https://www.researchgate.net/profile/Andres-Sevtsuk/publication/280218425_Location_and_Agglomeration_The_Distribution_of_Retail_and_Food_Businesses_in_Dense_Urban_Environments/links/58c6a34b92851c0ccbff6c1e/Location-and-Agglomeration-The-Distribution-of-Retail-and-Food-Businesses-in-Dense-Urban-Environments.pdf
7. AR2Net: An Attentive Neural Approach for Business Location Selection with Satellite Data and Urban Data: https://dl.acm.org/doi/abs/10.1145/3372406
8. Identification of robust retailing location patterns with complex network approaches: https://link.springer.com/content/pdf/10.1007/s40747-021-00335-8.pdf
9. KnowSite: Leveraging Urban Knowledge Graph for Site Selection: https://fi.ee.tsinghua.edu.cn/~dingjingtao/papers/KnowSite-Sigspatial23.pdf
10. Geo-spotting: Mining Online Location-based Services for Optimal Retail Store Placement: https://arxiv.org/abs/1306.1704

# Data

At minimum, we would need retailer and consumer data, as well as some location data for such analysis. We found some publicly available data to get us started: 

1. Kaggle Consumer Behavior and Shopping Habits Dataset: https://www.kaggle.com/datasets/zeesolver/consumer-behavior-and-shopping-habits-dataset
2. Kaggle Retail Data Analytics: https://www.kaggle.com/datasets/manjeetsingh/retaildataset
3. Foursquare Open Source Places: A new foundational dataset for the geospatial community: https://location.foursquare.com/resources/blog/products/foursquare-open-source-places-a-new-foundational-dataset-for-the-geospatial-community/
4. Data.gov Warehouse and Retail Sales: https://catalog.data.gov/dataset/warehouse-and-retail-sales
5. Data.gov Quarterly Retail Sales Tax Data by County and City (Iowa): https://catalog.data.gov/dataset/quarterly-retail-sales-tax-data-by-county-and-city
6. AWS Marketplace Places Data for Retail, Location & Marketing - Free Sample: https://aws.amazon.com/marketplace/pp/prodview-uheekokilsn5k#offers
7. Consumer Behavior and Shopping Habits Dataset: https://www.kaggle.com/datasets/zeesolver/consumer-behavior-and-shopping-habits-dataset
8. AWS Retail, Location & Marketing Data: https://aws.amazon.com/marketplace/search/results?category=2d1c8868-2642-4741-af5d-9ced77da7024&FULFILLMENT_OPTION_TYPE=DATA_EXCHANGE&filters=FULFILLMENT_OPTION_TYPE
9. Data.gov Retail Food Stores: https://catalog.data.gov/dataset/retail-food-stores
10. Intelligent Event data sample - Non attended events, Seattle: https://aws.amazon.com/marketplace/pp/prodview-xyzxww7esepcu?sr=0-20&ref_=beagle&applicationId=AWSMPContessa#overview
11. B2bConnect™ - Business Firmographics (Demo/Sample): https://aws.amazon.com/marketplace/pp/prodview-vxwy2qp2qp7tm?sr=0-50&ref_=beagle&applicationId=AWSMPContessa#offers
12. Free/Open Access AWS Retail, Location & Marketing Data: https://aws.amazon.com/marketplace/search/results?category=2d1c8868-2642-4741-af5d-9ced77da7024&FULFILLMENT_OPTION_TYPE=DATA_EXCHANGE&CONTRACT_TYPE=OPEN_DATA_LICENSES&filters=FULFILLMENT_OPTION_TYPE%2CCONTRACT_TYPE
13. AWS Market Place - Demand Intelligence - Global Intelligent Event Data API: https://aws.amazon.com/marketplace/pp/prodview-33rgkq2mboads?sr=0-26&ref_=beagle&applicationId=AWSMPContessa#overview
14. AWS Market Place - Spending Power™ (Demo/Sample): https://aws.amazon.com/marketplace/pp/prodview-hgng3yvvveg46?sr=0-43&ref_=beagle&applicationId=AWSMPContessa#offers
15. Quarterly Financial Report: Large U.S. Retail Trade Corporations, Third Quarter 2024: https://www.census.gov/econ/qfr/retail/current/index.html
16. CapitalOne Shopping Retail Statistics: https://capitaloneshopping.com/research/retail-statistics/
17. Sales Tax Collections Data | USA | 2019 - 2023: https://aws.amazon.com/marketplace/pp/prodview-wqbcwel53bwbg?sr=0-12&ref_=beagle&applicationId=AWSMPContessa#offers
18. Business Data US / 55M Businesses in the US ( sample): https://aws.amazon.com/marketplace/pp/prodview-ux4lwqysgqshg?sr=0-22&ref_=beagle&applicationId=AWSMPContessa#dataSets

# Modeling

The essential initial methodology is to look at this as a regression problem where we have the dependent variable as 
as the optimal location coordinates (GPS, Geolocation) and the explanatory or control variables include relevant 
radius demographics, income levels of potential customers, competitive retailer factors, target or expected revenue, 
location amenities (i.e., other businesses or social locations that affect traffic), etc. However, given that 
we would have sales data only for locations with relevant retail shops, it may be more fitting to run the regression model
with Expected Sales as the dependent variable and it's relative location as one of the dependent variables. Expectedly, machine learning would also 
help to model the complexity of the data and factors. 

Essentially, we would need sales data of retailer shop at a location, incomes and other demographics
(gender, education levels, employment statuses and industry)  of the population geo-distributions within the vicinity of the shop. 
We would also need data of other amenities in the shop's vicinity i.e., what 
other shops or facilities in the shop's vicinity could enable positive spillover effects in terms of shoppers traffic? Thus, the 
categories and patronage of nearby shops and amenities would also influence the traffic and revenue to a shop.
Competitive effects of similar or alternative category shops would also be captured in the model by including location
facility categories in the model. It also seems that including retail chain or franchise id would be enough to disambiguate
between cannibalization and competitive effects.