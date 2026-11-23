"""
AI Farming Advisor Module
Provides comprehensive agronomic cultivation guides for all 22 crops,
soil health analysis, environmental risk assessment, fertilizer calculator,
and an intelligent conversational assistant engine.
"""

# Agricultural Knowledge Base for all 22 Crops
CROP_KNOWLEDGE_BASE = {
    "rice": {
        "display_name": "Rice (Paddy)",
        "category": "Cereal Grain",
        "season": "Kharif (June - November)",
        "duration": "110 - 140 days",
        "soil_type": "Clayey loam, Deep alluvial soils with good water retention",
        "water_req": "High (1200 - 2000 mm); standing water (2-5 cm) required in vegetative phase",
        "ideal_npk": {"N": 90, "P": 42, "K": 43},
        "fertilizer": "Urea (N) + DAP (P) + MOP (K). Apply N in 3 split doses (basal, tillering, panicle initiation).",
        "irrigation": "Maintain saturated to standing water up to 10 days before harvest, then drain.",
        "pests_diseases": "Stem borer, Brown planthopper, Blast disease. Treat with Neem oil or Chlorantraniliprole.",
        "expected_yield": "3.5 - 6.0 tonnes / hectare",
        "tips": [
            "Use certified high-yield hybrid or SRI (System of Rice Intensification) methods.",
            "Incorporate green manure (Dhaincha/Sunhemp) prior to puddling to improve soil organic matter.",
            "Maintain 2-3 cm water level during tillering to suppress weed germination.",
            "Drain field completely 10-14 days before harvest to ensure uniform ripening."
        ]
    },
    "maize": {
        "display_name": "Maize (Corn)",
        "category": "Cereal Grain",
        "season": "Kharif / Rabi / Spring",
        "duration": "90 - 115 days",
        "soil_type": "Well-drained fertile loamy to silty loam soils; neutral pH (6.0 - 7.5)",
        "water_req": "Moderate (500 - 750 mm); highly sensitive to waterlogging",
        "ideal_npk": {"N": 80, "P": 48, "K": 20},
        "fertilizer": "High nitrogen feeder. Apply 1/3 N + all P & K at sowing, remaining N at knee-high and tasseling stages.",
        "irrigation": "Critical irrigation stages: Knee-high, Silking, and Grain filling stage. Avoid standing water.",
        "pests_diseases": "Fall Armyworm, Stem borer, Leaf blight. Spray Bacillus thuringiensis (Bt) or Emamectin Benzoate.",
        "expected_yield": "5.0 - 8.0 tonnes / hectare (Hybrid)",
        "tips": [
            "Maintain optimal plant spacing: 60 cm row-to-row, 20 cm plant-to-plant.",
            "Install pheromone traps early to monitor and control Fall Armyworm.",
            "Ensure ridge-and-furrow planting to prevent moisture stagnation during rains.",
            "Harvest when husk leaves turn straw yellow and black layer forms at grain base."
        ]
    },
    "chickpea": {
        "display_name": "Chickpea (Gram / Chana)",
        "category": "Pulse / Legume",
        "season": "Rabi (October - March)",
        "duration": "95 - 120 days",
        "soil_type": "Well-drained sandy loam to clay loam; tolerates light soil",
        "water_req": "Low (250 - 400 mm); thrives in cool, dry climate",
        "ideal_npk": {"N": 40, "P": 67, "K": 80},
        "fertilizer": "Leguminous (nitrogen-fixing). Apply Rhizobium & PSB bio-fertilizer seed treatment + basal DAP.",
        "irrigation": "1-2 light irrigations at pre-flowering and pod development. Never irrigate during peak flowering.",
        "pests_diseases": "Pod borer (Helicoverpa armigera), Fusarium wilt. Use pheromone traps and Trichoderma viride.",
        "expected_yield": "1.8 - 2.5 tonnes / hectare",
        "tips": [
            "Treat seeds with Rhizobium culture to enhance root nodulation and atmospheric nitrogen fixation.",
            "Nip the apical shoot (branch topping) at 30-40 days to induce profuse lateral branching.",
            "Avoid over-irrigation as it induces excessive vegetative growth and increases wilt incidence.",
            "Intercrop with mustard or linseed to deter pod borers naturally."
        ]
    },
    "kidneybeans": {
        "display_name": "Kidney Beans (Rajma)",
        "category": "Pulse / Legume",
        "season": "Rabi (Plains) / Kharif (Hills)",
        "duration": "100 - 130 days",
        "soil_type": "Rich, well-drained loamy soil with rich organic content",
        "water_req": "Moderate (400 - 600 mm); cannot tolerate moisture stress or water stagnation",
        "ideal_npk": {"N": 20, "P": 67, "K": 20},
        "fertilizer": "Unlike other pulses, lacks native nodulation; requires moderate basal N (60-80 kg/ha) and high P.",
        "irrigation": "Water every 10-12 days; key stages are vegetative branching and pod enlargement.",
        "pests_diseases": "Bean anthracnose, Whitefly, Rust. Spray Mancozeb or Neem formulation.",
        "expected_yield": "1.5 - 2.2 tonnes / hectare",
        "tips": [
            "Apply well-rotted Farmyard Manure (FYM) 10-15 tonnes/ha before sowing.",
            "Ridge planting is strongly recommended to protect delicate root systems from waterlogging.",
            "Harvest when pods turn yellowish-brown and seeds detach easily inside pods."
        ]
    },
    "pigeonpeas": {
        "display_name": "Pigeonpeas (Red Gram / Arhar / Tur)",
        "category": "Pulse / Legume",
        "season": "Kharif (June - December/January)",
        "duration": "150 - 180 days (Long duration) / 120 days (Early)",
        "soil_type": "Deep well-drained loam to heavy black soils",
        "water_req": "Moderate (600 - 800 mm); deep taproot provides high drought tolerance",
        "ideal_npk": {"N": 20, "P": 67, "K": 20},
        "fertilizer": "Basal application of 20 kg N, 50 kg P2O5, 20 kg K2O per hectare. Seed inoculation with Rhizobium.",
        "irrigation": "Irrigate at branching, flowering, and pod formation if dry spells exceed 20 days.",
        "pests_diseases": "Pod fly, Pod borer, Sterility mosaic virus. Spray Spinosad or Indoxacarb.",
        "expected_yield": "1.8 - 3.0 tonnes / hectare",
        "tips": [
            "Ideal for intercropping with Soybean, Cotton, or Sorghum (1:2 or 1:4 ratio).",
            "Deep ploughing in summer breaks hardpan and eradicates pupating pests.",
            "Pruning or nipping at 45 days encourages heavy bushiness and higher pod clusters."
        ]
    },
    "mothbeans": {
        "display_name": "Moth Beans (Matki)",
        "category": "Arid Pulse / Legume",
        "season": "Kharif (July - October)",
        "duration": "75 - 90 days",
        "soil_type": "Sandy, light arid soils; thrives even in nutrient-poor desert soils",
        "water_req": "Low (200 - 450 mm); one of the most drought-hardy legumes",
        "ideal_npk": {"N": 20, "P": 48, "K": 20},
        "fertilizer": "Minimal nutrient requirement; 10 kg N + 30 kg P2O5 / ha as basal dose.",
        "irrigation": "Usually grown under rainfed conditions; 1 life-saving irrigation at pod fill if rains fail.",
        "pests_diseases": "Yellow mosaic virus, Jassids. Spray Neem kernel extract or Imidacloprid.",
        "expected_yield": "0.8 - 1.5 tonnes / hectare",
        "tips": [
            "Acts as excellent soil cover crop, preventing wind and water soil erosion in dry zones.",
            "Seeds should be treated with Thiram or Captan @ 2.5 g/kg seed before sowing.",
            "Short duration allows farmers to take second catch crops if late showers occur."
        ]
    },
    "mungbean": {
        "display_name": "Mung Bean (Green Gram)",
        "category": "Short Pulse / Legume",
        "season": "Kharif / Summer (March - May)",
        "duration": "60 - 75 days (Very Fast)",
        "soil_type": "Well-drained fertile loamy to sandy loam soils",
        "water_req": "Low to Moderate (350 - 500 mm); sensitive to water stagnation",
        "ideal_npk": {"N": 20, "P": 48, "K": 20},
        "fertilizer": "Apply 15-20 kg N and 40 kg P2O5/ha; inoculate with Rhizobium phaseoli.",
        "irrigation": "2-3 irrigations during summer crop; critical at pre-flowering and pod development.",
        "pests_diseases": "Powdery mildew, Whiteflies (vector of Yellow Mosaic). Spray Carbendazim / Dimethoate.",
        "expected_yield": "1.0 - 1.6 tonnes / hectare",
        "tips": [
            "Fits perfectly in rice-wheat rotations as a 60-day summer cash and green manure crop.",
            "Pick pods in 2-3 flushes as they ripen to prevent shattering losses.",
            "Foliar spray of 2% DAP at 30 days significantly boosts pod counts."
        ]
    },
    "blackgram": {
        "display_name": "Black Gram (Urad Dal)",
        "category": "Pulse / Legume",
        "season": "Kharif / Spring / Summer",
        "duration": "70 - 85 days",
        "soil_type": "Heavier soils, loamy black soils, alluvial soils with good moisture retention",
        "water_req": "Moderate (400 - 600 mm); thrives in warm, humid weather",
        "ideal_npk": {"N": 40, "P": 67, "K": 19},
        "fertilizer": "20 kg N + 40 kg P2O5 + 20 kg K2O / ha. Sulphur application (20 kg/ha) boosts protein synthesis.",
        "irrigation": "Provide irrigation at 20 days (branching) and 45 days (pod formation). Avoid excess water.",
        "pests_diseases": "Leaf crinkle virus, Cercospora leaf spot. Use disease-resistant certified varieties.",
        "expected_yield": "1.2 - 1.8 tonnes / hectare",
        "tips": [
            "Seed treatment with Trichoderma viride protects against root rot.",
            "Harvest promptly when 80-85% pods turn black to avoid seed shattering.",
            "Stubble incorporation after harvest adds up to 30 kg nitrogen equivalent per hectare."
        ]
    },
    "lentil": {
        "display_name": "Lentil (Masoor)",
        "category": "Pulse / Legume",
        "season": "Rabi (October - March)",
        "duration": "110 - 130 days",
        "soil_type": "Wide range: light loams to heavy clay; tolerant to mild soil salinity",
        "water_req": "Low (300 - 450 mm); excellent cool season crop",
        "ideal_npk": {"N": 20, "P": 67, "K": 19},
        "fertilizer": "20 kg N + 40 kg P2O5 per ha. Responsive to Zinc and Boron application.",
        "irrigation": "Requires 1-2 light irrigations; excessive moisture promotes vegetative wilt.",
        "pests_diseases": "Rust, Wilt, Aphids. Foliar spray of wettable sulphur @ 2 g/L.",
        "expected_yield": "1.4 - 2.2 tonnes / hectare",
        "tips": [
            "Can be sown as relay crop (Utera/Paira) in standing rice 10 days before paddy harvest.",
            "Requires fine tilth and moisture at sowing for uniform germination.",
            "Harvest in early morning hours to prevent dry pod shattering."
        ]
    },
    "pomegranate": {
        "display_name": "Pomegranate (Anar)",
        "category": "Fruit Orchard",
        "season": "Perennial / Bahar treatment (Mrig, Hasta, Ambe Bahar)",
        "duration": "Orchard takes 2-3 years to bear; fruit matures 135-165 days after blossom",
        "soil_type": "Deep well-drained loamy to sandy loam soils; tolerant to light alkalinity (pH 6.5 - 8.2)",
        "water_req": "Moderate (500 - 700 mm); drip irrigation strongly recommended",
        "ideal_npk": {"N": 20, "P": 20, "K": 40},
        "fertilizer": "Annual adult tree: 600g N, 200g P2O5, 200g K2O + 20 kg FYM per plant per year in split fertigation.",
        "irrigation": "Water stress (withholding water) is used to regulate Bahar flowering, followed by regular drip.",
        "pests_diseases": "Bacterial blight (Teliya), Fruit borer (butterfly). Copper Oxychloride + Streptocycline sprays.",
        "expected_yield": "12 - 18 tonnes / hectare (from mature 4th year onwards)",
        "tips": [
            "Practice sanitary orchard management: remove and destroy all mummified fruits.",
            "Bag fruits with butter paper bags 60 days before maturity to prevent sun-scald and borer attacks.",
            "Regulate canopy by annual pruning after harvest to maximize sunlight penetration."
        ]
    },
    "banana": {
        "display_name": "Banana",
        "category": "Fruit Crop",
        "season": "Planted year-round (June-July or October-November preferred)",
        "duration": "10 - 13 months",
        "soil_type": "Deep, rich, well-drained loamy soil with high organic matter (pH 6.5 - 7.5)",
        "water_req": "High (1500 - 2200 mm); continuous moisture needed, drip fertigation best",
        "ideal_npk": {"N": 100, "P": 75, "K": 50},
        "fertilizer": "Heavy feeder: 200g N, 60g P2O5, 300g K2O per plant in 4-5 split doses throughout growth cycle.",
        "irrigation": "Irrigate every 3-4 days in summer, 7-8 days in winter. Never let soil dry to cracking.",
        "pests_diseases": "Sigatoka leaf spot, Panama wilt, Rhizome weevil. Propiconazole spray and sucker treatment.",
        "expected_yield": "60 - 85 tonnes / hectare (Tissue Culture Grand Naine)",
        "tips": [
            "Use disease-free, virus-indexed Tissue Culture suckers for uniform flowering and high tonnage.",
            "Perform desuckering regularly, allowing only 1 follower sucker to develop near harvest.",
            "Provide bamboo / nylon propping to support heavy bunch weight during winds."
        ]
    },
    "mango": {
        "display_name": "Mango",
        "category": "Fruit Orchard",
        "season": "Perennial; flowering Dec-Feb, harvest April-July",
        "duration": "Tree life 40+ years; fruit takes 110-130 days from fruit set",
        "soil_type": "Deep, well-drained alluvial, red loamy soils; depth > 2 meters without hard sub-pan",
        "water_req": "Moderate (700 - 1000 mm); withhold irrigation 2 months prior to flowering",
        "ideal_npk": {"N": 20, "P": 25, "K": 30},
        "fertilizer": "Bearing tree: 1000g N, 500g P2O5, 1000g K2O per tree applied after harvest in August.",
        "irrigation": "Irrigate fortnightly during fruit development stage (pea size to marble size to mature).",
        "pests_diseases": "Mango hopper, Powdery mildew, Anthracnose. Spray Imidacloprid and Hexaconazole.",
        "expected_yield": "8 - 15 tonnes / hectare",
        "tips": [
            "Spray Potassium Nitrate (1%) in October to break shoot dormancy and induce profuse panicles.",
            "Prune center branches after harvest to allow 360-degree sunlight inside tree canopy.",
            "Harvest with 1 cm stalk intact to prevent sap burn blemishes on fruit skin."
        ]
    },
    "grapes": {
        "display_name": "Grapes",
        "category": "Fruit Orchard / Vine",
        "season": "Perennial; Foundation pruning (April) and Forward pruning (October)",
        "duration": "120 - 145 days from October pruning to harvest",
        "soil_type": "Well-drained sandy loam to clay loam; excellent drainage mandatory",
        "water_req": "Moderate (500 - 700 mm); drip irrigation under Bower/Y-trellis system",
        "ideal_npk": {"N": 20, "P": 134, "K": 200},
        "fertilizer": "High Potassium feeder: Fertigation with Water Soluble Fertilizers (19:19:19, 0:52:34, 0:0:50).",
        "irrigation": "Carefully metered drip irrigation; stop irrigation 7-10 days prior to harvest to increase Brix (sugar).",
        "pests_diseases": "Downy mildew, Powdery mildew, Mealybugs. Regular prophylactic spray of Bordeaux mixture.",
        "expected_yield": "20 - 30 tonnes / hectare",
        "tips": [
            "Use Gibberellic Acid (GA3) dips at precise berry stages for bunch elongation and berry sizing.",
            "Girdling canes 60 days after pruning boosts uniform berry sugar accumulation.",
            "Maintain open canopy architecture on trellis to reduce morning leaf moisture and mildew."
        ]
    },
    "watermelon": {
        "display_name": "Watermelon",
        "category": "Cucurbit / Fruit",
        "season": "Zaid / Summer (January - April)",
        "duration": "80 - 100 days",
        "soil_type": "Sandy loam, riverbed alluvial soils with rapid warm drainage (pH 6.0 - 7.0)",
        "water_req": "Moderate (400 - 600 mm); regular moisture during fruit growth, dry at ripening",
        "ideal_npk": {"N": 100, "P": 18, "K": 50},
        "fertilizer": "Apply 100 kg N, 50 kg P2O5, 80 kg K2O/ha. K promotes sweetness and thick rind.",
        "irrigation": "Drip irrigation with silver-black plastic mulch. Reduce water 10 days before harvest.",
        "pests_diseases": "Fruit fly, Powdery mildew, Fusarium wilt. Install Cue-lure pheromone traps.",
        "expected_yield": "35 - 50 tonnes / hectare (Hybrid)",
        "tips": [
            "Silver-black plastic mulching boosts soil temperature, retains moisture, and stops weed growth.",
            "Introduce bee boxes (1-2 hives/acre) to ensure 100% fruit setting via pollination.",
            "Tap fruit to check maturity: a dull, hollow thud indicates ripe sweet fruit."
        ]
    },
    "muskmelon": {
        "display_name": "Muskmelon (Cantaloupe)",
        "category": "Cucurbit / Fruit",
        "season": "Zaid / Summer (February - May)",
        "duration": "70 - 90 days",
        "soil_type": "Fertile sandy loam with good organic matter and high aeration",
        "water_req": "Low to Moderate (350 - 500 mm); prefers hot, dry sunny weather for sugar sweetness",
        "ideal_npk": {"N": 100, "P": 18, "K": 50},
        "fertilizer": "80 kg N, 60 kg P2O5, 60 kg K2O / ha. Top dress with Potassium Sulphate for aroma and TSS.",
        "irrigation": "Irrigate at 5-7 days interval; do not wet vines and fruits directly to prevent rotting.",
        "pests_diseases": "Downy mildew, Aphids, Leaf miners. Spray Neem oil or Mancozeb.",
        "expected_yield": "18 - 25 tonnes / hectare",
        "tips": [
            "Hot dry climate with long sunshine hours produces the sweetest fruits with high Brix > 11%.",
            "Harvest at 'full slip' stage (when the stem separates smoothly with slight thumb pressure).",
            "Avoid excessive nitrogen during fruit ripening to prevent rind splitting."
        ]
    },
    "apple": {
        "display_name": "Apple",
        "category": "Temperate Fruit Orchard",
        "season": "Temperate perennial; requires 800 - 1200 chilling hours (< 7°C)",
        "duration": "Tree life 30+ years; fruit ripens in 130 - 160 days from bloom (July - October)",
        "soil_type": "Deep, well-aerated, loamy soil rich in organic humus; pH 5.5 - 6.8",
        "water_req": "Moderate to High (1000 - 1250 mm evenly distributed throughout the season)",
        "ideal_npk": {"N": 20, "P": 134, "K": 200},
        "fertilizer": "Annual adult tree: 500g N, 250g P2O5, 750g K2O. High Boron and Calcium requirement.",
        "irrigation": "Critical irrigation periods: flowering to fruit set and fruit enlargement phase.",
        "pests_diseases": "Apple scab, Woolly apple aphid, Powdery mildew. Captan or Difenoconazole spray schedule.",
        "expected_yield": "15 - 25 tonnes / hectare",
        "tips": [
            "Maintain 15-20% pollinizer varieties (e.g. Golden Delicious, Granny Smith) in the orchard.",
            "Spray Calcium Chloride (0.5%) before harvest to prevent 'bitter pit' disorder in storage.",
            "Practice annual dormant winter pruning to encourage fruiting spurs."
        ]
    },
    "orange": {
        "display_name": "Orange (Citrus / Mandarin)",
        "category": "Citrus Fruit Orchard",
        "season": "Sub-tropical perennial; Ambe / Mrig Bahar harvests",
        "duration": "Takes 240 days from fruit set to maturity",
        "soil_type": "Deep, well-drained sandy loam or alluvial soil; free from calcium carbonate and hard sub-pan",
        "water_req": "Moderate (800 - 1100 mm); drip irrigation prevents collar rot",
        "ideal_npk": {"N": 20, "P": 16, "K": 10},
        "fertilizer": "Adult tree: 600g N, 200g P2O5, 300g K2O + Micronutrient foliar spray (Zn, Fe, Mn, Cu, B).",
        "irrigation": "Drip irrigation under tree basin; keep water away from direct contact with trunk collar.",
        "pests_diseases": "Citrus canker, Citrus psylla (vector of Greening disease), Phytophthora gummosis.",
        "expected_yield": "15 - 22 tonnes / hectare",
        "tips": [
            "Paint tree trunk with Bordeaux paste (1:1:10) twice a year to prevent fungal foot rot.",
            "Apply Zinc Sulphate (0.5%) foliar spray twice a year to cure citrus mottle leaf.",
            "Harvest with clippers leaving small button; never pull fruit directly."
        ]
    },
    "papaya": {
        "display_name": "Papaya",
        "category": "Tropical Fruit",
        "season": "Planted year-round (February-March or July-August)",
        "duration": "9 - 11 months to first harvest; productive for 2-3 years",
        "soil_type": "Rich, well-drained loamy to alluvial soil; absolute zero tolerance to waterlogging",
        "water_req": "Moderate to High (1000 - 1400 mm); constant moderate moisture",
        "ideal_npk": {"N": 50, "P": 60, "K": 50},
        "fertilizer": "250g N, 250g P2O5, 500g K2O per plant per year in 6 bimonthly split applications.",
        "irrigation": "Ring system or drip irrigation; strictly avoid wetting trunk stem.",
        "pests_diseases": "Papaya ringspot virus (PRSV), Collar rot, Root knot nematode. Grow Red Lady 786 hybrid.",
        "expected_yield": "60 - 100 tonnes / hectare (Gynodioecious Hybrids)",
        "tips": [
            "Plant on raised beds (30 cm height) to ensure rapid rainwater runoff during downpours.",
            "Remove all lateral side shoots continuously to direct all energy to central fruiting column.",
            "Harvest when color break appears (green skin turns 10-20% yellow at fruit base)."
        ]
    },
    "coconut": {
        "display_name": "Coconut",
        "category": "Plantation / Palm",
        "season": "Perennial; continuous year-round monthly harvests",
        "duration": "Begins yielding in 4-6 years; commercial life 60-80 years",
        "soil_type": "Coastal sandy loam, alluvial, red sandy loam with good depth and permeability",
        "water_req": "High (1500 - 2500 mm); loves warm humid tropical coastal climate",
        "ideal_npk": {"N": 20, "P": 16, "K": 30},
        "fertilizer": "Adult palm: 500g N, 320g P2O5, 1200g K2O + 1 kg Magnesium Sulphate + 50 kg green manure annually.",
        "irrigation": "Drip irrigation: 50-70 liters per palm per day during dry months.",
        "pests_diseases": "Rhinoceros beetle, Red palm weevil, Bud rot. Place naphthalene balls in top leaf axils.",
        "expected_yield": "80 - 140 nuts / palm / year (8,000 - 14,000 nuts/ha)",
        "tips": [
            "Apply common salt (1-2 kg/palm/year) in non-coastal inland soils to enhance chlorine uptake.",
            "Grow intercrops like Cocoa, Pepper, Banana, or Pineapple in the wide 7.5 x 7.5m tree spaces.",
            "Maintain soil mulch around palm basins to preserve root zone moisture in summer."
        ]
    },
    "cotton": {
        "display_name": "Cotton (White Gold)",
        "category": "Fiber / Cash Crop",
        "season": "Kharif (April - June sowing, Oct - Feb picking)",
        "duration": "160 - 190 days",
        "soil_type": "Deep black cotton soils (Vertisols) or fertile alluvial soils with high moisture retention",
        "water_req": "Moderate (650 - 900 mm); requires dry sunny weather during boll opening",
        "ideal_npk": {"N": 120, "P": 46, "K": 20},
        "fertilizer": "120 kg N, 60 kg P2O5, 60 kg K2O / ha. Apply N in 3 splits (sowing, square formation, flowering).",
        "irrigation": "Irrigate at critical stages: squaring, flowering, and boll development. Cease when bolls burst.",
        "pests_diseases": "Pink bollworm, Whitefly, Bacterial blight. Install Pheromone traps and use Bt hybrids.",
        "expected_yield": "2.5 - 4.0 tonnes / hectare (Seed cotton)",
        "tips": [
            "De-top (nipping terminal bud) at 80-90 days to check excessive vegetative growth and enhance bolls.",
            "Foliar spray of 2% Potassium Nitrate (KNO3) during boll filling improves fiber length and strength.",
            "Pick cotton only after morning dew evaporates to ensure clean, high-grade dry lint."
        ]
    },
    "jute": {
        "display_name": "Jute (Golden Fiber)",
        "category": "Fiber / Commercial",
        "season": "Kharif (March - July)",
        "duration": "120 - 140 days",
        "soil_type": "New alluvial soils along river basins and delta plains",
        "water_req": "High (1200 - 1600 mm); loves warm humid weather (>80% humidity)",
        "ideal_npk": {"N": 80, "P": 46, "K": 40},
        "fertilizer": "60 kg N, 30 kg P2O5, 30 kg K2O / ha. Top dress urea in two splits at 3 and 6 weeks.",
        "irrigation": "Needs early pre-sowing showers or irrigation; thrives under subsequent monsoon rains.",
        "pests_diseases": "Stem rot, Yellow mite, Semilooper. Spray Neem formulations or Mancozeb.",
        "expected_yield": "2.8 - 3.8 tonnes / hectare (Dry fiber)",
        "tips": [
            "Harvest at the small pod stage (120 days) for the best balance of fiber strength and fine fineness.",
            "Retting water should be clean and slow-moving; use microbial retting inoculants (e.g. CRIJAF Sona).",
            "Sun-dry extracted fiber thoroughly before baling to prevent fiber staining and fungal decay."
        ]
    },
    "coffee": {
        "display_name": "Coffee (Arabica & Robusta)",
        "category": "Beverage / Plantation",
        "season": "Perennial; Blossom showers March-April, Harvest Nov-Feb",
        "duration": "First harvest at 3-4 years; berries take 8-9 months from flowering to harvest",
        "soil_type": "Deep, well-drained, porous red or lateritic loamy forest soils rich in humus (pH 6.0 - 6.5)",
        "water_req": "High (1500 - 2500 mm); requires distinct dry period for flower bud rest followed by blossom rain",
        "ideal_npk": {"N": 100, "P": 30, "K": 30},
        "fertilizer": "Apply N:P2O5:K2O in balanced proportion (120:90:120 kg/ha) in split doses pre-monsoon and post-monsoon.",
        "irrigation": "Sprinkler irrigation (Blossom and Backing showers) during March-April guarantees 100% fruit set.",
        "pests_diseases": "Coffee berry borer, White stem borer, Leaf rust (Hemileia vastatrix). Spray Bordeaux mixture 0.5%.",
        "expected_yield": "1.0 - 1.8 tonnes / hectare (Clean coffee beans)",
        "tips": [
            "Grow under two-tier shade trees (temporary shade like Dadap and permanent shade like Silver Oak).",
            "Harvest only fully ripe bright red cherries for superior cup quality and premium market price.",
            "Mulch coffee basin with leaf litter to preserve soil humidity and stimulate feeder root growth."
        ]
    }
}


def get_crop_details(crop_label: str) -> dict:
    """
    Returns rich agronomic advice for a given crop name.
    Falls back to a safe default if crop is not recognized.
    """
    crop_clean = str(crop_label).strip().lower()
    if crop_clean in CROP_KNOWLEDGE_BASE:
        return CROP_KNOWLEDGE_BASE[crop_clean]

    return {
        "display_name": crop_label.capitalize(),
        "category": "Agricultural Crop",
        "season": "Seasonal Kharif / Rabi",
        "duration": "90 - 120 days",
        "soil_type": "Well-drained fertile loamy soil",
        "water_req": "Moderate (500 - 800 mm)",
        "ideal_npk": {"N": 50, "P": 50, "K": 50},
        "fertilizer": "Apply balanced NPK (19:19:19) with organic compost.",
        "irrigation": "Irrigate according to topsoil moisture status.",
        "pests_diseases": "Inspect foliage weekly for chewing insects and fungal spots.",
        "expected_yield": "2.0 - 4.0 tonnes / hectare",
        "tips": [
            "Conduct soil test before planting to calibrate fertilizer dosages.",
            "Use certified high-germination seed stocks.",
            "Ensure proper field drainage to avoid moisture stagnation."
        ]
    }


def analyze_soil_health(N: float, P: float, K: float, ph: float) -> dict:
    """
    Analyzes soil nutrient parameters against standard agricultural benchmarks.
    Returns status indicators, score, and prescriptive corrections.
    """
    findings = []
    recommendations = []
    score = 100

    # Nitrogen (kg/ha)
    if N < 40:
        findings.append({"parameter": "Nitrogen (N)", "status": "Deficient", "color": "red",
                         "detail": f"{N:.1f} kg/ha (Critically Low)"})
        recommendations.append("Apply Neem-coated Urea or well-rotted cow dung manure to boost vegetative vigor.")
        score -= 20
    elif N > 110:
        findings.append({"parameter": "Nitrogen (N)", "status": "Excessive", "color": "orange",
                         "detail": f"{N:.1f} kg/ha (High)"})
        recommendations.append("Reduce nitrogenous fertilizers to avoid vegetative lodging and increased pest vulnerability.")
        score -= 10
    else:
        findings.append({"parameter": "Nitrogen (N)", "status": "Optimum", "color": "green",
                         "detail": f"{N:.1f} kg/ha (Healthy Range)"})

    # Phosphorus (kg/ha)
    if P < 35:
        findings.append({"parameter": "Phosphorus (P)", "status": "Deficient", "color": "red",
                         "detail": f"{P:.1f} kg/ha (Low)"})
        recommendations.append("Apply Single Super Phosphate (SSP) or DAP to stimulate root establishment.")
        score -= 15
    elif P > 90:
        findings.append({"parameter": "Phosphorus (P)", "status": "High", "color": "orange",
                         "detail": f"{P:.1f} kg/ha (High)"})
        recommendations.append("Phosphorus reserve is ample; withhold additional phosphate fertilizers.")
        score -= 5
    else:
        findings.append({"parameter": "Phosphorus (P)", "status": "Optimum", "color": "green",
                         "detail": f"{P:.1f} kg/ha (Healthy Range)"})

    # Potassium (kg/ha)
    if K < 35:
        findings.append({"parameter": "Potassium (K)", "status": "Deficient", "color": "red",
                         "detail": f"{K:.1f} kg/ha (Low)"})
        recommendations.append("Apply Muriate of Potash (MOP) to enhance disease resistance and grain filling.")
        score -= 15
    elif K > 120:
        findings.append({"parameter": "Potassium (K)", "status": "High", "color": "orange",
                         "detail": f"{K:.1f} kg/ha (High)"})
        recommendations.append("High potassium detected; ensure Magnesium and Calcium balance.")
        score -= 5
    else:
        findings.append({"parameter": "Potassium (K)", "status": "Optimum", "color": "green",
                         "detail": f"{K:.1f} kg/ha (Healthy Range)"})

    # Soil pH
    if ph < 5.5:
        findings.append({"parameter": "Soil pH", "status": "Strongly Acidic", "color": "red",
                         "detail": f"{ph:.2f} (Nutrient lockup likely)"})
        recommendations.append("Apply Agricultural Agricultural Lime (CaCO3) or dolomite @ 1-2 tonnes/ha to neutralize acidity.")
        score -= 25
    elif ph > 8.0:
        findings.append({"parameter": "Soil pH", "status": "Alkaline / Saline", "color": "red",
                         "detail": f"{ph:.2f} (Micro-nutrient deficiency likely)"})
        recommendations.append("Apply Agricultural Gypsum (CaSO4) and incorporate organic farmyard manure or green manure.")
        score -= 25
    elif 6.0 <= ph <= 7.5:
        findings.append({"parameter": "Soil pH", "status": "Ideal Neutral", "color": "green",
                         "detail": f"{ph:.2f} (Maximum nutrient bioavailability)"})
    else:
        findings.append({"parameter": "Soil pH", "status": "Slightly Off-Neutral", "color": "orange",
                         "detail": f"{ph:.2f} (Acceptable for most crops)"})
        score -= 5

    score = max(20, min(100, score))

    return {
        "score": score,
        "rating": "Excellent" if score >= 85 else ("Good" if score >= 70 else ("Fair" if score >= 50 else "Poor")),
        "findings": findings,
        "recommendations": recommendations
    }


def analyze_environmental_risks(temperature: float, humidity: float, rainfall: float, ph: float) -> list:
    """
    Evaluates micro-climatic weather conditions and generates actionable risk warnings.
    """
    risks = []

    # Temperature Alerts
    if temperature > 38.0:
        risks.append({
            "type": "Extreme Heat Stress",
            "level": "High",
            "icon": "🔥",
            "message": f"Ambient temperature ({temperature:.1f}°C) exceeds heat tolerance for most non-arid crops. Provide frequent light mulched irrigations to prevent flower drop."
        })
    elif temperature < 15.0:
        risks.append({
            "type": "Low Temperature / Chilling",
            "level": "Moderate",
            "icon": "❄️",
            "message": f"Low temperature ({temperature:.1f}°C) may retard germination and slow vegetative growth. Suitable for temperate & Rabi crops (apple, lentil, chickpea)."
        })

    # Rainfall Alerts
    if rainfall < 45.0:
        risks.append({
            "type": "Drought / Low Precipitation Risk",
            "level": "High",
            "icon": "☀️",
            "message": f"Precipitation ({rainfall:.1f} mm) is severely deficient. Supplementary drip or furrow irrigation is mandatory to sustain crops."
        })
    elif rainfall > 260.0:
        risks.append({
            "type": "Excess Water / Flooding Hazard",
            "level": "High",
            "icon": "🌧️",
            "message": f"Heavy rainfall ({rainfall:.1f} mm) may cause soil saturation and root asphyxiation. Ensure unhindered surface drainage canals. High suitability for Rice/Jute."
        })

    # Humidity Alerts
    if humidity > 88.0:
        risks.append({
            "type": "Fungal & Bacterial Disease Risk",
            "level": "Moderate",
            "icon": "🍄",
            "message": f"High relative humidity ({humidity:.1f}%) promotes blast, powdery mildew, and blight spores. Maintain plant spacing and prepare prophylactic bio-fungicide sprays."
        })
    elif humidity < 35.0:
        risks.append({
            "type": "Atmospheric Desiccation",
            "level": "Moderate",
            "icon": "💨",
            "message": f"Dry atmospheric conditions ({humidity:.1f}% humidity) accelerate evapotranspiration. Protect sensitive seedlings with windbreaks or mulch."
        })

    # Soil pH extremes
    if ph < 5.2 or ph > 8.2:
        risks.append({
            "type": "Root Nutrient Lockup Risk",
            "level": "High",
            "icon": "⚗️",
            "message": f"Extreme pH ({ph:.2f}) inhibits root absorption of Phosphorus, Iron, and Zinc even if present in soil."
        })

    if not risks:
        risks.append({
            "type": "Optimal Agronomic Conditions",
            "level": "Low",
            "icon": "🌿",
            "message": "Weather, moisture, and soil conditions are well-balanced with no acute stress factors detected."
        })

    return risks


def calculate_fertilizer_recommendation(crop_name: str, N: float, P: float, K: float) -> dict:
    """
    Computes fertilizer requirements to bridge the gap between current soil NPK
    and benchmark targets for the recommended crop.
    Converts nutrient deficits to commercial fertilizers:
      - Urea (46% N)
      - DAP (18% N, 46% P2O5)
      - MOP (60% K2O)
    """
    details = get_crop_details(crop_name)
    target_npk = details.get("ideal_npk", {"N": 60, "P": 40, "K": 40})

    diff_n = max(0.0, target_npk["N"] - N)
    diff_p = max(0.0, target_npk["P"] - P)
    diff_k = max(0.0, target_npk["K"] - K)

    # Fertilizer calculation:
    # 1. Supply P using DAP (18% N, 46% P2O5)
    dap_req = (diff_p / 0.46) if diff_p > 0 else 0.0
    n_from_dap = dap_req * 0.18

    # 2. Supply remaining N using Urea (46% N)
    remaining_n = max(0.0, diff_n - n_from_dap)
    urea_req = (remaining_n / 0.46) if remaining_n > 0 else 0.0

    # 3. Supply K using MOP (60% K2O)
    mop_req = (diff_k / 0.60) if diff_k > 0 else 0.0

    return {
        "crop": details["display_name"],
        "current_npk": {"N": N, "P": P, "K": K},
        "target_npk": target_npk,
        "deficit": {"N": diff_n, "P": diff_p, "K": diff_k},
        "fertilizer_quantities_kg_ha": {
            "Urea (46% N)": round(urea_req, 1),
            "DAP (18% N, 46% P)": round(dap_req, 1),
            "Muriate of Potash (60% K)": round(mop_req, 1)
        },
        "bags_estimate_50kg": {
            "Urea Bags": round(urea_req / 50, 1),
            "DAP Bags": round(dap_req / 50, 1),
            "MOP Bags": round(mop_req / 50, 1)
        }
    }


def ask_farming_assistant(user_query: str, context: dict = None) -> str:
    """
    Rule and heuristic based AI Farming Assistant that answers agronomic questions
    regarding soil management, pests, fertilizers, crop rotation, and seasonal scheduling.
    """
    q = user_query.lower().strip()

    # Contextual awareness if recent prediction is passed
    crop_ctx = context.get("crop", "") if context else ""
    soil_ctx = context.get("soil", {}) if context else {}

    # Responses based on intent matching
    if any(w in q for w in ["ph", "acidic", "alkaline", "lime", "gypsum"]):
        return (
            "🧪 **Soil pH Management Advice:**\n\n"
            "• **Acidic Soil (pH < 6.0):** Add agricultural limestone (calcium carbonate) or dolomite at 1-2 tonnes per hectare. "
            "Wood ash also helps raise pH while adding potassium.\n"
            "• **Alkaline Soil (pH > 7.5):** Apply agricultural gypsum (calcium sulphate) and incorporate organic matter like farmyard manure or compost. "
            "Elemental sulphur can also be applied for persistent alkalinity.\n"
            "• **Ideal Range:** Most crops thrive between pH 6.0 and 7.2 where phosphorus, nitrogen, and micronutrients are most bio-available."
        )

    if any(w in q for w in ["drought", "dry", "less water", "save water", "irrigation"]):
        return (
            "💧 **Smart Water & Irrigation Management:**\n\n"
            "• **Drip Irrigation:** Saves 40-60% water compared to flood irrigation and delivers nutrients directly to the root zone.\n"
            "• **Mulching:** Apply organic crop residue (straw, dry leaves) or silver-black plastic mulch (25-30 microns) to reduce evaporation by up to 70%.\n"
            "• **Drought-Tolerant Crops:** Consider mothbeans, chickpeas, pigeonpeas, or pearl millet if water availability is unpredictable.\n"
            "• **Timing:** Irrigate in early mornings or evenings to minimize heat evaporation losses."
        )

    if any(w in q for w in ["fertilizer", "urea", "dap", "npk", "potash", "manure", "organic"]):
        return (
            "🌱 **Fertilizer & Soil Nutrition Guidance:**\n\n"
            "• **Nitrogen (N):** Essential for vegetative growth and lush green foliage. Apply in split doses (basal, tillering, flowering) to prevent leaching.\n"
            "• **Phosphorus (P):** Critical for robust root establishment and early seedling vigor. Always apply as a basal dose near the root zone.\n"
            "• **Potassium (K):** Enhances disease tolerance, drought resistance, and fruit/grain quality. Split doses work best in sandy soils.\n"
            "• **Organic Enrichment:** Add 5-10 tonnes/hectare of composted Farmyard Manure (FYM) or Vermicompost before planting to revitalize beneficial soil microbes."
        )

    if any(w in q for w in ["pest", "insect", "disease", "fungus", "spray", "borer", "blight"]):
        return (
            "🐛 **Integrated Pest Management (IPM):**\n\n"
            "• **Cultural Control:** Practice crop rotation, deep summer ploughing to expose hibernating pupae, and maintain clean field borders.\n"
            "• **Biological Control:** Use Neem oil (1500-3000 ppm) spray as a safe repellent. Release Trichogramma parasitoids for caterpillar control.\n"
            "• **Physical Traps:** Install yellow sticky traps (for aphids and whiteflies) and pheromone traps (5-8 per hectare) for early detection of borers.\n"
            "• **Chemical Sprays:** Use certified systemic insecticides or fungicides only when pest threshold crosses economic injury levels (EIL)."
        )

    if any(w in q for w in ["rotation", "intercrop", "sequence"]):
        return (
            "🔄 **Crop Rotation & Intercropping Strategies:**\n\n"
            "• **Cereal-Legume Rotation:** Rotate heavy nitrogen consumers (Rice, Maize, Wheat) with nitrogen-fixing pulses (Chickpea, Mungbean, Blackgram).\n"
            "• **Deep vs. Shallow Roots:** Alternate deep-rooted crops (Cotton, Pigeonpea) with shallow-rooted crops (Onion, Leafy vegetables, Rice) to tap different soil strata.\n"
            "• **Pest Cycle Disruption:** Never plant solanaceous or cucurbit crops consecutively in the same plot to avoid building nematode and viral loads."
        )

    if any(w in q for w in ["yield", "maximize", "profit", "increase production"]):
        return (
            "📈 **Strategies to Maximize Farm Yield & Profits:**\n\n"
            "1. **Quality Seed:** Always use certified hybrid or treated seed stocks with >95% germination rate.\n"
            "2. **Precision Spacing:** Maintain optimal plant population per acre; overcrowding creates microclimates favoring fungal pathogens.\n"
            "3. **Timely Sowing:** Planting within the optimal meteorological window ensures the flowering stage escapes extreme heat or monsoon dry spells.\n"
            "4. **Foliar Micronutrients:** Spray Zinc Sulphate (0.5%) and Borax (0.2%) during early flowering to prevent flower drop."
        )

    # If asking about the currently recommended crop
    if crop_ctx and crop_ctx in q:
        details = get_crop_details(crop_ctx)
        return (
            f"🌾 **Specific Agronomic Advice for {details['display_name']}:**\n\n"
            f"• **Category:** {details['category']} ({details['season']})\n"
            f"• **Growth Cycle:** {details['duration']}\n"
            f"• **Ideal Soil:** {details['soil_type']}\n"
            f"• **Irrigation:** {details['irrigation']}\n"
            f"• **Expected Yield:** {details['expected_yield']}\n"
            f"• **Key Advice:** {details['tips'][0]} {details['tips'][1]}"
        )

    # General fallback with helpful agricultural tips
    return (
        "🤖 **AI Farming Assistant Response:**\n\n"
        "Here are key recommendations tailored to sustainable farming:\n"
        "1. **Soil Testing:** Test your soil every 2 seasons to calibrate fertilizer applications and avoid unnecessary chemical expenses.\n"
        "2. **Water Efficiency:** Adopt micro-irrigation (drip or micro-sprinkler) to reduce weed pressure and conserve groundwater.\n"
        "3. **Weather Forecast Monitoring:** Check 7-day meteorological forecasts before scheduling fertilizer applications or pesticide sprays.\n"
        "4. **Soil Organic Carbon:** Maintain soil organic carbon above 0.5% by recycling crop residues and applying green manure.\n\n"
        "💡 *You can ask me about: Soil pH management, Fertilizer calculation, Drought mitigation, Pest control, or specific crop advice!*"
    )
