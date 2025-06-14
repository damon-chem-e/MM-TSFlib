# Product Object

- Keepa-01 - (0) Object

Keepa-01
175.7 Feb 2016

## About

The product object contains all of our price history data and basic product information.

## Returned by

The product object is returned by the following requests:
- Product Searches (US)
- Product Request

## Important

Always evaluate the productType field first. This field determines what data for the product is available.

## Format

| Field | Description |
|-------|-------------|
| productType | Most always be evaluated first. Determines what data for the product is available. Possible values: STANDARD, UNTRACKABLE, EBOOK, INACCESSIBLE, INVALID, VARIATION_PARENT |
| asin | The ASIN of the product. Example: B000980QVG3W |
| domainId | The domainId of the product's Amazon locale. Possible values: 1: com, 3: co.uk, 4: de, 5: fr, 6: co.jp, 7: ca, 8: it, 9: es, 10: in, 11: com.mx |
| title | Title of the product. Example: Canon PowerShot GX400 Digital Camera with 30x Optical Zoom (Black) |
| trackingSince | States the time we have started tracking this product, in Keepa Time minutes. Example: 2711219 |
| listedSince | States the time the item was first listed on Amazon, in Keepa Time minutes. Example: 2711219 |
| lastUpdate | States the last time we have updated the information for this product, in Keepa Time minutes. Example: 2711219 |
| lastRatingUpdate | States the last time we have updated the product rating and review count, in Keepa Time minutes. Example: 2711219 |
| lastPriceChange | States the last time we have registered a price change (any price type), in Keepa Time minutes. Example: 2711219 |
| lastCheapUpdate | States the last time we have updated the eBay prices for this product, in Keepa Time minutes. Example: 2711219 |
| lastStockUpdate | The most recent update of the stock data for this product's offers, in Keepa Time minutes. Example: 2711219 |
| images | Provides metadata for images associated with the product. Example: `{"l": "libXlux.RyRL.jpg", "h": 1000, "w": 1000, "m": "glXlibdolmp1.jpg", "mh": 500, "mw": 500}` |
| rootCategory | The root category of the product. Example: 002086 |
| categories | Array of Amazon category IDs. Example: [000604] |
| categoryTree | Represents the category tree as an ordered array of objects, each consisting of two fields: `catId` and `name`. |
| parentAsin | The ASIN of the parent product (if the product has variations, otherwise null). Example: B000980QVG3W |
| parentAsinHistory | The history of the parentAsin field. Example: [2711319, B000980QVG3W] |
| variations | Contains up to 4000 variations of this product. Example: `{"asin": "B0070796E0", "attributes": [{"dimension": "Size", "value": "7.26 LB"}, {"dimension": "CheeseName", "value": "Cheddar"}]}` |
| videos | Provides metadata for videos associated with the product. Example: `{"title": "Compressed Air Duster", "image": "334b-07b-5.jpg", "duration": 45, "creator": "seller", "name": "Innovation", "url": "https://m.media-amazon.com/images/S/..."}` |
| aplus | Provides enhanced A+ content from Amazon product listings. Example: `{"module": [{"text": ["Experience premium quality and design.", "Engineered for performance and style."], "image": ["https://m.media-amazon.com/images/S/..."], "imageAltText": ["Healthy delicious recipes"], "video": ["https://m.media-amazon.com/images/S/..."]}], "fromManufacturer": true}` |
| frequentlyBoughtTogether | One or two "Frequently Bought Together" ASINs. Example: ["B00M0QVG3M", "B00T85FMWY"] |
| eanList | A list of EANs assigned to this product. Example: ["580809624952"] |
| upcList | A list of UPCs assigned to this product. Example: ["545496590086"] |
| gtinList | A list of GTINs assigned to this product. Example: ["09914141008012"] |
| manufacturer | Name of the manufacturer. Example: Canon |
| brand | An item's brand. Example: Sony |
| brandStoreName | The store name of the item's brand. Example: Hot Wheels |
| brandStoreUrl | The brand store URL path. Example: /s/store/LEGO/page/017EF856-865D-4B56-A171-EA61CAFF45DD |
| brandStoreUrlName | The brand store name from the URL path. Example: LEGO |
| productGroup | The item's productGroup. Example: Camera |
| websiteDisplayGroup | A categorization of products that behave similarly. Example: apparel_display_on_website |
| websiteDisplayGroupName | A categorization name of products that behave similarly. Example: apparel |
| salesRankDisplayGroup | The categorization of products for which the primary sales rank is based on. Example: apparel_display_on_website |
| type | The item's productType. Example: VIDEO_GAME_CONTROLLER |
| partNumber | The item's partNumber. Example: DSC-H300/BM-RB |
| binding | The item's binding. Example: Hardcover |
| scent | The scent of the product. Example: Lavender |
| shortDescription | A brief description of the product. Example: A soothing lavender-scented candle. |
| activeIngredients | Active ingredients present in the product. Example: Lavender essential oil, Soy wax |
| specialIngredients | Special ingredients used in the product that may have unique properties. Example: Beeswax blend, Natural dyes |
| itemForm | The form or physical state of the item. Example: Liquid |
| itemTypeKeyword | Keywords describing the type or category of the item. Example: body-billions |
| recommendedUsesForProduct | Recommended uses for the product to guide customers. Example: Aromatherapy, Home Decoration |
| pattern | The pattern or design featured on the product. Example: Striped, Floral |
| specificUsesForProduct | Specific uses for the product, providing detailed applications. Example: ["Relaxation", "Decoration"] |
| businessDiscount | The highest business discount percentage. Example: 14 |
| lastBusinessDiscountUpdate | States the last time we have updated the businessDiscount field, in Keepa Time minutes. Example: 2711219 |
| safetyWarning | Safety warnings associated with the product to inform users of potential hazards. Example: Keep away from open flames. |
| packageWidth | The package's width in millimeters. Example: 144 |
| packageWeight | The package's weight in grams. Example: 1500 |
| packageQuantity | Quantity of items in the package. Example: 4 |
| itemHeight | The item's height in millimeters. Example: 144 |
| itemLength | The item's length in millimeters. Example: 144 |
| itemWidth | The item's width in millimeters. Example: 144 |
| itemWeight | The item's weight in grams. Example: 1500 |
| availabilityAmazon | Availability of the Amazon offer. Possible values: -1, 0, 1, 2, 3, 4 |
| availabilityAmazonDelay | If availabilityAmazon has the value 4, this field will provide the delay interval in hours. Example: [24, 48] |
| buyBoxEligibleOfferCounts | If buyBoxEligibleOffersCount is available, it represents an array of integers. Example: [0, 1, 2, 3, 4, 5, 6, 7] |
| competitivePriceThreshold | The competitivePriceThreshold field is a price based on competitive prices from other retailers. |
| suggestedLowerPrice | The suggested lower price of the item, including shipping. |
| ebayListingIds | Contains the lowest priced matching eBay listing IDs. Example: [273344496193, 0] |
| isAdultProduct | Indicates if the item is considered to be for adults only. Example: true |
| isHeatSensitive | Indicates if the item is heat sensitive. Example: true |
| isMerchOnDemand | True if this product is an Amazon Merch on Demand product. Example: true |
| isHaul | True if this product is an Amazon Haul product. Example: true |
| ingredients | The ingredient list of the product. Example: Purified Carbonated Water, Natural Flavors |
| launchpad | Indicates if the item is listed in the launchpad category. Example: true |
| audienceRating | Audience rating. Example: PG-13 |
| urlSlug | The product listing URL slug. Example: Ring-Video-Doorbell-Satin-Nickel-2020-Release |
| returnRate | This field provides information about the customer return rate for a product. Possible values: null, 1, 2 |
| newPriorityMAP | Whether or not the lowest new price is restricted by MAP. Example: true |
| isEligibleForTradeIn | Whether or not the product is eligible for trade-in. Example: true |
| isEligibleForSuperSaverShipping | Whether or not the product's buy box is eligible for free shipping. Example: true |
| fbaFees | Contains an object providing the FBA pick & pack fees of this product. Example: `{"LastUpdate": keepaTime, "pickPackFee": 209}` |
| variableClosingFee | The variable closing fee. Example: 81 |
| referralFeePercentage | The Amazon seller referral fee percent. Example: 12 |
| coupon | Contains coupon details for the buy box offer of the product. Example: [200, -15] |
| couponHistory | Contains historical values for the coupon field. Example: [2711319, 200, -15, ...] |
| promotions | Contains an array of active promotions. Example: `{"type": "SNS", "amount": 100, "discountPercent": 10, "snsBulkDiscountPercent": 5}` |
| formats | For books only: An array listing other available formats or bindings of a book. Example: `{"asin": "B0070796E0", "format": "Kindle"}` |
| unitCount | Object that contains unit count data. Example: `{"unitValue": 72.5, "unitType": "oz", "eachUnitCount": 12}` |
| stats | Optional field, set only if the stats parameter was used in the Product Request. |
| salesRankReference | The category node id of the main sales rank. Example: 281052 |
| salesRankReferenceHistory | A long array containing the historical category node id(s) of the main sales rank. Example: [2711319, 281052, ...] |
| salesRanks | An object containing sales rank histories. Example: `{"281052": [keepaTime, salesRank, ...], "123112042": [keepaTime, salesRank, ...]}` |
| lastSoldUpdate | States the last time we have updated the monthlySold field, in Keepa Time minutes. Example: 2711219 |
| monthlySold | How often the product was bought in the past month. |
| monthlySoldHistory | Contains historical values of the monthlySold field. |
| rentalDetails | Contains the rental details description of the rental buy box offer. Example: Rented from: Seller Rental Price One Semester: $25.24 |
| rentalSellerId | Contains the seller id of the rental buy box offer. Example: A2LT7EE7UG99Q |
| rentalPrices | This object contains the rental prices. Example: `{"initialPrice": 2524, "shortExtnPrice": 1261, "longExtnPrice": 2524, "fullPrice": 3489}` |
| reviews | This object contains variation specific review and rating counts histories. Example: `{"lastUpdate": keepaTime, "ratingCount": [keepaTime, ratingCount, ...], "reviewCount": [keepaTime, reviewCount, ...]}` |
| offers | Optional field, set only if the offers parameter was used in the Product Request. |
| liveOffersOrder | Optional field, set only if the offers parameter was used in the Product Request. Example: [2, 0, 2, 18, 15] |
| buyBoxSellerIdHistory | Optional field, set only if the offers or buybox parameter was used in the Product Request. Example: [2860626, ATVPDKIKX0DER, ...] |
| buyBoxUsedHistory | Optional field, set only if the offers or buybox parameter was used in the Product Request. Example: [2860626, ATVPDKIKX0DER, "4", "1", ...] |
| isRedirectASIN | Valid only if the offers parameter was used in the Product Request. Example: true |
| isSNS | Boolean indicating if the product's Buy Box is available for subscribe and save. Example: true |
| offersSuccessful | Valid only if the offers parameter was used in the Product Request. Example: true |
| cw | A two-dimensional history array containing the product's historical data. |
| stats | Optional field, set only if the stats parameter was used in the Product Request. |
| salesRankReference | The category node id of the main sales rank. Example: 281052 |
| salesRankReferenceHistory | A long array containing the historical category node id(s) of the main sales rank. Example: [2711319, 281052, ...] |
| salesRanks | An object containing sales rank histories. Example: `{"281052": [keepaTime, salesRank, ...], "123112042": [keepaTime, salesRank, ...]}` |
| lastSoldUpdate | States the last time we have updated the monthlySold field, in Keepa Time minutes. Example: 2711219 |
| monthlySold | How often the product was bought in the past month. |
| monthlySoldHistory | Contains historical values of the monthlySold field. |
| rentalDetails | Contains the rental details description of the rental buy box offer. Example: Rented from: Seller Rental Price One Semester: $25.24 |
| rentalSellerId | Contains the seller id of the rental buy box offer. Example: A2LT7EE7UG99Q |
| rentalPrices | This object contains the rental prices. Example: `{"initialPrice": 2524, "shortExtnPrice": 1261, "longExtnPrice": 2524, "fullPrice": 3489}` |
| reviews | This object contains variation specific review and rating counts histories. Example: `{"lastUpdate": keepaTime, "ratingCount": [keepaTime, ratingCount, ...], "reviewCount": [keepaTime, reviewCount, ...]}` |
| offers | Optional field, set only if the offers parameter was used in the Product Request. |
| liveOffersOrder | Optional field, set only if the offers parameter was used in the Product Request. Example: [2, 0, 2, 18, 15] |
| buyBoxSellerIdHistory | Optional field, set only if the offers or buybox parameter was used in the Product Request. Example: [2860626, ATVPDKIKX0DER, ...] |
| buyBoxUsedHistory | Optional field, set only if the offers or buybox parameter was used in the Product Request. Example: [2860626, ATVPDKIKX0DER, "4", "1", ...] |
| isRedirectASIN | Valid only if the offers parameter was used in the Product Request. Example: true |
| isSNS | Boolean indicating if the product's Buy Box is available for subscribe and save. Example: true |
| offersSuccessful | Valid only if the offers parameter was used in the Product Request. Example: true |
| csv | A two-dimensional history array containing the product's historical data. See below for details. |

## CSV

A two-dimensional history array containing the product's historical data. Access the first dimension index using the following enum/constants:

- 0 - AMAZON: Amazon price history
- 1 - NEW: Marketplace New price history
- 2 - USED: Marketplace Used price history
- 3 - SALES: Sales Rank history (not every product has a Sales Rank; variation items usually don't have individual sales ranks)
- 4 - LISTPRICE: List Price history
- 5 - COLLECTIBLE: Collectible price history
- 6 - REFURBISHED: Refurbished price history
- 7 - NEW_FBM_SHIPPING: 3rd party (not including Amazon) New price history including shipping costs, only fulfilled by merchant (FBM)
- 8 - LIGHTNING_DEAL: Lightning Deal price history (special, relevant information below)
- 9 - WAREHOUSE: Amazon Warehouse price history
- 10 - NEW_FBA: Price history of the lowest 3rd party (not including Amazon/Warehouse) New offer that is fulfilled by Amazon
- 11 - COUNT_NEW: New offer count history (= count of marketplace merchants selling the product as new)
- 12 - COUNT_USED: Used offer count history
- 13 - COUNT_REFURBISHED: Refurbished offer count history
- 14 - COUNT_COLLECTIBLE: Collectible offer count history
- 15 - EXTRA_INFO_UPDATES: History of past updates to all offers-parameter related data: offers, isSNS, isRedirectASIN, and the csv types NEW_FBM_SHIPPING, WAREHOUSE, NEW_FBA, RATING, COUNT_REVIEWS, USED_SHIPPING, COLLECTIBLE_SHIPPING, BUY_BOX_USED_SHIPPING, PRIME_EXCL and REFURBISHED_SHIPPING. As updates to those fields are infrequent, it is essential to know when our system updated them. The absolute value indicates the number of offers fetched at the given time. If the value is positive, it means all available offers were fetched. If negative, there were more offers than fetched.
- 16 - RATING: The product's rating history (an integer from 0 to 50, e.g., 45 = 4.5 stars)
- 17 - COUNT_REVIEWS: The product's rating count history
- 18 - BUY_BOX_SHIPPING: The New buy box price history, including shipping costs. If no offer qualified for the buy box (or if the buy box is a used offer), the price has the value -1
- 19 - USED_NEW_SHIPPING: "Used - Like New" price history, including shipping costs
- 20 - USED_VERY_GOOD_SHIPPING: "Used - Very Good" price history, including shipping costs
- 21 - USED_GOOD_SHIPPING: "Used - Good" price history, including shipping costs
- 22 - USED_ACCEPTABLE_SHIPPING: "Used - Acceptable" price history, including shipping costs
- 23 - COLLECTIBLE_NEW_SHIPPING: "Collectible - Like New" price history, including shipping costs
- 24 - COLLECTIBLE_VERY_GOOD_SHIPPING: "Collectible - Very Good" price history, including shipping costs
- 25 - COLLECTIBLE_GOOD_SHIPPING: "Collectible - Good" price history, including shipping costs
- 26 - COLLECTIBLE_ACCEPTABLE_SHIPPING: "Collectible - Acceptable" price history, including shipping costs
- 27 - REFURBISHED_SHIPPING: Refurbished price history, including shipping costs
- 28 - EBAY_NEW_SHIPPING: Price history of the lowest new price on the respective eBay locale, including shipping costs
- 29 - EBAY_USED_SHIPPING: Price history of the lowest used price on the respective eBay locale, including shipping costs
- 30 - TRADE_IN: Trade-in price history (Amazon trade-in is not available for every locale)
- 31 - RENTAL: Rental price history (requires the use of the rental and offers parameters; Amazon Rental is only available for Amazon US)
- 32 - BUY_BOX_USED_SHIPPING: The Used buy box price history (any sub-condition), including shipping costs. If no offer qualified for the used buy box, the price has the value -1
- 33 - PRIME_EXCL: Price history of the lowest Prime exclusive New offer

The second dimension contains the price/value history in the format Keepa time minutes, value, [...] or, if the type includes *SHIPPING costs, the format Keepa time minutes, price, shipping costs, [...]. If history is unavailable, it is null. The price is an integer of the respective Amazon locale's smallest currency unit (e.g., euro cents or yen). If no offer was available in the given interval (e.g., out of stock), the price has the value -1.

### Important CSV Information:

- We only append a new entry to the history array if the price/value changed, not every time we update the product.
- Shipping and Handling costs are not included unless specified.
- Amazon is considered to be part of the marketplace. If Amazon has the overall lowest new price, the marketplace new price in the corresponding time interval is identical to the Amazon price.
- The following types are only set if the offers parameter was used in the Product Request: NEW_FBM_SHIPPING, WAREHOUSE, NEW_FBA, RATING, COUNT_REVIEWS, BUY_BOX_SHIPPING, USED_*_SHIPPING, COLLECTIBLE_*_SHIPPING, REFURBISHED_SHIPPING, BUY_BOX_USED_SHIPPING, and PRIME_EXCL.
- About Lightning deals: If a deal is currently active, the last entry of the history array will contain the end date (a future date) with a price of -1. When accessing the current deal price, handle this special case. Alternatively, use the current price array of the stats object to access the current price. If no deal is currently active but the last entry has the value -1 with a date in the future, it means an upcoming deal is announced (scheduled to start at that date).
- About eBay prices: eBay listings often have incorrect information and/or product codes, causing item mismatches and incorrect pricing information. Use eBay price history information with caution and do not rely on their accuracy. We update eBay prices at different intervals than Amazon prices.
- We can add new data types to the csv field from time to time without announcement. Do not expect a fixed size of the first array dimension in your implementation.


## Keepa Time Minutes

Time format used for all timestamps. To convert to an uncompressed Unix epoch time, add 21564000 to the Keepa time value before converting:

- **For milliseconds:** `(keepaTime + 21564000) * 60000`
- **For seconds:** `(keepaTime + 21564000) * 60`

This adjustment aligns Keepa's internal time with the standard Unix epoch.

## Updates

| Date | Update Description |
|------|-------------------|
| 25 days later | Videos and aPlus added. |
| 1 month later | Added new field: snsBulkDiscountPercent to the promotions field. |
| 10 days later | Added new fields: gtinList, formula, imageAltText (part of the aPlus object). |
| 1 month later | Added new fields: images, websiteDisplayGroupName, websiteDisplayGroup. Removed fields: imagesCSV. |
| 18 days later | Added new field: salesRankDisplayGroup. Increased the limit of variations from 1800 to 4000. |
| 8 days later | The variation specific ratingCount history is no longer updated, as that data point was removed by Amazon. |
