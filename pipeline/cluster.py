# Note: Add scikit-learn>=1.5.0 to pipeline/requirements.txt if not already present.

import json
import logging
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.cluster import HDBSCAN
from pydantic import BaseModel, ConfigDict

from pipeline.normalize import Athlete, Hometown, OlympicParalympicStatus

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

EARTH_RADIUS_KM = 6371.0
EPSILON_RADIANS = 50.0 / EARTH_RADIUS_KM

CITY_TO_STATE = {
    # Northeast
    "New York City": "NY", "Brooklyn": "NY", "Queens": "NY",
    "Bronx": "NY", "Manhattan": "NY", "Staten Island": "NY",
    "Yonkers": "NY", "Albany": "NY", "Buffalo": "NY", "Rochester": "NY",
    "Syracuse": "NY", "Long Island": "NY", "Mineola": "NY",
    "Oyster Bay": "NY", "Suffern": "NY", "Port Jefferson": "NY",
    "Massapequa": "NY",
    "Newark": "NJ", "Jersey City": "NJ", "Hoboken": "NJ",
    "Paterson": "NJ", "Trenton": "NJ", "Passaic": "NJ",
    "Hackensack": "NJ", "Cranford": "NJ", "Freehold": "NJ",
    "South Plainfield": "NJ", "Berwyn": "IL", "Wayne": "NJ",
    "Union City": "NJ",
    "Philadelphia": "PA", "Pittsburgh": "PA", "Reading": "PA",
    "Scranton": "PA", "Bryn Mawr": "PA", "Phillipsburg": "NJ",
    "Bridgeton": "NJ", "Allentown": "PA", "Spruce Hill": "PA",
    "Boston": "MA", "Cambridge": "MA", "Worcester": "MA",
    "Springfield": "MA", "Marlborough": "MA", "Wenham": "MA",
    "Medford": "MA", "Stoneham": "MA", "Haverhill": "MA",
    "Wareham": "MA", "Fitchburg": "MA",
    "Providence": "RI", "Manchester": "NH", "Portsmouth": "NH",
    "Hanover": "NH", "Londonderry": "NH",
    "Hartford": "CT", "New Haven": "CT", "Bridgeport": "CT",
    "Burlington": "VT",
    "Portland": "ME",
    "Washington, D.C.": "DC",
    "Baltimore": "MD", "Frederick": "MD", "Silver Spring": "MD",
    "Mount Airy": "MD", "La Plata": "MD",
    # Midwest
    "Chicago": "IL", "Champaign": "IL", "Downers Grove": "IL",
    "Melrose Park": "IL", "Palos Park": "IL", "Gurnee": "IL",
    "Detroit": "MI", "Ann Arbor": "MI", "Lansing": "MI",
    "Saginaw": "MI", "Grand Haven": "MI", "Allegan": "MI",
    "Royal Oak": "MI", "Lapeer": "MI", "Eaton Rapids": "MI",
    "Warren": "MI", "Clinton Township": "MI",
    "Cleveland": "OH", "Columbus": "OH", "Cincinnati": "OH",
    "Akron": "OH", "Toledo": "OH", "Canton": "OH", "Dayton": "OH",
    "Upper Arlington": "OH", "Celina": "OH",
    "Indianapolis": "IN", "Muncie": "IN", "Goshen": "IN",
    "Terre Haute": "IN", "New Albany": "IN",
    "Milwaukee": "WI", "Madison": "WI", "Green Bay": "WI",
    "Waukesha": "WI", "Sheboygan": "WI", "Oconto": "WI",
    "Grantsburg": "WI",
    "Minneapolis": "MN", "Saint Paul": "MN", "St. Paul": "MN",
    "Duluth": "MN", "Mankato": "MN", "Lakeville": "MN",
    "Owatonna": "MN", "Litchfield": "MN",
    "St. Louis": "MO", "Kansas City": "MO",
    "Omaha": "NE", "Lincoln": "NE",
    "Des Moines": "IA", "Davenport": "IA", "Council Bluffs": "IA",
    "Cresco": "IA", "Larchwood": "IA", "Schuyler": "IA",
    "Topeka": "KS", "Wichita": "KS", "Salina": "KS", "Lawrence": "KS",
    "Cape Girardeau": "MO", "Wyaconda": "MO",
    "Fargo": "ND", "Sioux Falls": "SD",
    # South
    "Atlanta": "GA", "Macon": "GA", "Cartersville": "GA",
    "Clarkesville": "GA", "Eastman": "GA",
    "Charlotte": "NC", "Raleigh": "NC", "Greensboro": "NC",
    "Hickory": "NC", "Huntersville": "NC", "Greenville": "NC",
    "Columbia": "SC", "Greenwood": "SC",
    "Jacksonville": "FL", "Miami": "FL", "Tampa": "FL", "Orlando": "FL",
    "Cape Canaveral": "FL", "Coral Gables": "FL", "Winter Haven": "FL",
    "Riverview": "FL",
    "Birmingham": "AL", "Mobile": "AL", "Huntsville": "AL",
    "Nashville": "TN", "Memphis": "TN", "Knoxville": "TN",
    "Chattanooga": "TN", "Kingsport": "TN",
    "Louisville": "KY",
    "New Orleans": "LA", "Baton Rouge": "LA", "Laurel": "MS",
    "Jackson": "MS",
    "Houston": "TX", "Dallas": "TX", "San Antonio": "TX",
    "Austin": "TX", "Fort Worth": "TX", "Plano": "TX",
    "Amarillo": "TX", "Wichita Falls": "TX", "Wylie": "TX",
    "Mesquite": "TX", "Terrell": "TX", "Stockton": "TX",
    "Perryton": "TX",
    "Oklahoma City": "OK", "Tulsa": "OK", "Stillwater": "OK",
    "Checotah": "OK", "Claremore": "OK", "Dewar": "OK",
    "Midwest City": "OK",
    "Little Rock": "AR", "Tuckerman": "AR",
    "Charleston": "SC",
    "Hopkins": "MN",
    "San Juan": "PR",
    # Mountain
    "Denver": "CO", "Boulder": "CO", "Colorado Springs": "CO",
    "Aspen": "CO", "Steamboat Springs": "CO", "Vail": "CO",
    "Berthoud": "CO", "Wheat Ridge": "CO", "Longmont": "CO",
    "Salt Lake City": "UT",
    "Phoenix": "AZ", "Tucson": "AZ", "Nogales": "AZ",
    "Prescott Valley": "AZ",
    "Albuquerque": "NM", "Silver City": "NM",
    "Boise": "ID", "Orofino": "ID",
    "Helena": "MT", "Billings": "MT", "Missoula": "MT",
    "Cheyenne": "WY",
    "Las Vegas": "NV", "Reno": "NV", "Las Vegas Valley": "NV",
    "Park Rapids": "MN",
    # Pacific / West
    "Los Angeles": "CA", "Inglewood": "CA", "Bell": "CA",
    "South Pasadena": "CA", "Glendale": "CA", "Santa Clarita": "CA",
    "Pasadena": "CA", "West Covina": "CA", "Fullerton": "CA",
    "Santa Maria": "CA", "Orange": "CA", "Hanford": "CA",
    "Fresno": "CA", "Merced": "CA", "Fremont": "CA",
    "San Francisco": "CA", "Oakland": "CA", "San Jose": "CA",
    "San Diego": "CA", "Riverside": "CA", "Sacramento": "CA",
    "Sunnyvale": "CA", "Los Gatos": "CA", "Harbor City": "CA",
    "San Gabriel": "CA",
    "Portland": "OR", "Eugene": "OR", "Beaverton": "OR",
    "Seattle": "WA", "Tacoma": "WA", "Spokane": "WA",
    "Redmond": "WA", "Everett": "WA", "Benton City": "WA",
    "Richland": "WA", "Woodinville": "WA",
    "Anchorage": "AK", "Juneau": "AK", "Palmer": "AK",
    "Honolulu": "HI", "Mililani": "HI", "Paia": "HI",
    "Falmouth": "MA",
}

DISPLAY_NAME_OVERRIDES = {
    # NYC metro - if medoid lands in any of these, label as NYC Metro
    "Union City": "New York Metro Region",
    "Bell": "Los Angeles Metro Region",
    "Yonkers": "New York Metro Region",
    "Bronx": "New York Metro Region",
    "Inglewood": "Los Angeles Metro Region",
    "Hoboken": "New York Metro Region",
    "Jersey City": "New York Metro Region",
    "Newark": "New York Metro Region",
    "Brooklyn": "New York Metro Region",
    "Queens": "New York Metro Region",
    "Cambridge": "Boston Metro Region",
    "Oakland": "Bay Area Region",
    "San Jose": "Bay Area Region",
    "Fremont": "Bay Area Region",
    "Pasadena": "Los Angeles Metro Region",
    "Glendale": "Los Angeles Metro Region",
}

# 42 culturally-authentic US regions, organized by macro-region.
# Each entry: (lat_min, lat_max, lon_min, lon_max, region_name,
# macro_region, do_not_group_with_list, tribal_nations_within)
# Order matters: more specific regions FIRST, broader fallbacks LAST.
# A point matches the FIRST region whose bbox contains it.
REGIONAL_CONTEXTS = [
    # --- TERRITORIES (most specific lat/lon, check first) ---
    {"name": "American Samoa", "macro": "Territories",
     "lat_min": -14.55, "lat_max": -14.10, "lon_min": -171.10, "lon_max": -169.40,
     "tribal": []},
    {"name": "Guam (Guåhan)", "macro": "Territories",
     "lat_min": 13.20, "lat_max": 13.70, "lon_min": 144.60, "lon_max": 145.00,
     "tribal": ["CHamoru"]},
    {"name": "Northern Mariana Islands", "macro": "Territories",
     "lat_min": 14.10, "lat_max": 20.55, "lon_min": 144.85, "lon_max": 146.10,
     "tribal": ["CHamoru", "Refaluwasch"]},
    {"name": "US Virgin Islands", "macro": "Territories",
     "lat_min": 17.65, "lat_max": 18.45, "lon_min": -65.10, "lon_max": -64.55,
     "tribal": []},
    {"name": "San Juan Metro", "macro": "Territories",
     "lat_min": 18.30, "lat_max": 18.55, "lon_min": -66.30, "lon_max": -65.85,
     "tribal": ["Taíno (ancestral)"]},
    {"name": "Puerto Rico", "macro": "Territories",
     "lat_min": 17.85, "lat_max": 18.55, "lon_min": -67.30, "lon_max": -65.20,
     "tribal": ["Taíno (ancestral)"]},

    # --- ALASKA (specific high-latitude) ---
    {"name": "Southeast Alaska / Inside Passage", "macro": "Alaska",
     "lat_min": 54.50, "lat_max": 60.00, "lon_min": -141.00, "lon_max": -130.00,
     "tribal": ["Tlingit", "Haida", "Tsimshian"]},
    {"name": "Southcentral Alaska", "macro": "Alaska",
     "lat_min": 59.00, "lat_max": 63.00, "lon_min": -154.00, "lon_max": -144.00,
     "tribal": ["Dena'ina Athabaskan", "Ahtna"]},
    {"name": "Interior Alaska", "macro": "Alaska",
     "lat_min": 63.00, "lat_max": 67.50, "lon_min": -150.00, "lon_max": -141.00,
     "tribal": ["Athabaskan"]},
    {"name": "Bush Alaska", "macro": "Alaska",
     "lat_min": 51.00, "lat_max": 71.50, "lon_min": -180.00, "lon_max": -154.00,
     "tribal": ["Yup'ik", "Iñupiat", "Aleut"]},

    # --- HAWAII ---
    {"name": "Oahu / Honolulu Metro", "macro": "Hawaii",
     "lat_min": 21.20, "lat_max": 21.80, "lon_min": -158.30, "lon_max": -157.60,
     "tribal": ["Kanaka Maoli (Native Hawaiian)"]},
    {"name": "Hawaii Neighbor Islands", "macro": "Hawaii",
     "lat_min": 18.90, "lat_max": 22.30, "lon_min": -160.30, "lon_max": -154.80,
     "tribal": ["Kanaka Maoli (Native Hawaiian)"]},

    # --- PACIFIC: California (3-tier per user spec) ---
    # Northern California: lat >= 37.0
    {"name": "Bay Area", "macro": "Pacific",
     "lat_min": 37.00, "lat_max": 38.85, "lon_min": -123.05, "lon_max": -121.55,
     "tribal": ["Ohlone (ancestral)"]},
    {"name": "Sacramento / NorCal Interior", "macro": "Pacific",
     "lat_min": 38.40, "lat_max": 42.00, "lon_min": -124.45, "lon_max": -119.85,
     "tribal": ["Maidu", "Wintun", "Yurok", "Hupa"]},
    {"name": "Northern California / Central Valley North", "macro": "Pacific",
     "lat_min": 37.00, "lat_max": 38.40, "lon_min": -121.55, "lon_max": -119.85,
     "tribal": ["Yokuts (ancestral)"]},
    # Central California: 35.0 <= lat < 37.0
    {"name": "Central California", "macro": "Pacific",
     "lat_min": 35.00, "lat_max": 37.00, "lon_min": -122.50, "lon_max": -118.95,
     "tribal": ["Yokuts (ancestral)", "Chumash (ancestral)"]},
    # Southern California: lat < 35.0 (Pico Act 35°47'N reference)
    {"name": "Greater Los Angeles", "macro": "Pacific",
     "lat_min": 33.30, "lat_max": 34.85, "lon_min": -119.45, "lon_max": -117.65,
     "tribal": ["Tongva (ancestral)", "Chumash (ancestral)"]},
    {"name": "Inland Empire", "macro": "Pacific",
     "lat_min": 33.40, "lat_max": 35.00, "lon_min": -117.65, "lon_max": -114.45,
     "tribal": ["Cahuilla", "Serrano", "Luiseño"]},
    {"name": "San Diego / Tijuana Borderlands", "macro": "Pacific",
     "lat_min": 32.50, "lat_max": 33.55, "lon_min": -117.65, "lon_max": -116.05,
     "tribal": ["Kumeyaay", "Luiseño"]},

    # --- PACIFIC NORTHWEST ---
    {"name": "Greater Seattle / Puget Sound", "macro": "Pacific",
     "lat_min": 46.85, "lat_max": 48.95, "lon_min": -123.20, "lon_max": -121.50,
     "tribal": ["Coast Salish", "Duwamish", "Suquamish", "Muckleshoot"]},
    {"name": "Greater Portland / Willamette Valley", "macro": "Pacific",
     "lat_min": 43.85, "lat_max": 46.30, "lon_min": -123.85, "lon_max": -121.85,
     "tribal": ["Confederated Tribes of Grand Ronde", "Confederated Tribes of Siletz"]},
    {"name": "Inland Northwest", "macro": "Pacific",
     "lat_min": 45.50, "lat_max": 49.00, "lon_min": -120.55, "lon_max": -115.85,
     "tribal": ["Spokane", "Coeur d'Alene", "Yakama", "Nez Perce"]},

    # --- MOUNTAIN WEST ---
    {"name": "Greater Las Vegas / Southern Nevada", "macro": "Mountain West",
     "lat_min": 35.05, "lat_max": 36.85, "lon_min": -115.85, "lon_max": -114.05,
     "tribal": ["Southern Paiute"]},
    {"name": "Northern Nevada / Reno-Tahoe", "macro": "Mountain West",
     "lat_min": 38.85, "lat_max": 41.95, "lon_min": -120.05, "lon_max": -117.05,
     "tribal": ["Washoe", "Northern Paiute"]},
    {"name": "Front Range Colorado", "macro": "Mountain West",
     "lat_min": 38.40, "lat_max": 40.85, "lon_min": -105.45, "lon_max": -104.45,
     "tribal": ["Ute (ancestral)", "Cheyenne (ancestral)", "Arapaho (ancestral)"]},
    {"name": "Western Slope Colorado", "macro": "Mountain West",
     "lat_min": 36.95, "lat_max": 41.00, "lon_min": -109.05, "lon_max": -105.45,
     "tribal": ["Ute Mountain Ute", "Southern Ute"]},
    {"name": "Wasatch Front", "macro": "Mountain West",
     "lat_min": 40.10, "lat_max": 41.40, "lon_min": -112.20, "lon_max": -111.55,
     "tribal": ["Ute", "Shoshone", "Goshute"]},
    {"name": "Mormon Corridor / Greater Utah", "macro": "Mountain West",
     "lat_min": 36.95, "lat_max": 42.00, "lon_min": -114.05, "lon_max": -109.05,
     "tribal": ["Ute", "Paiute", "Navajo (south UT)"]},
    {"name": "Big Sky Country", "macro": "Mountain West",
     "lat_min": 41.00, "lat_max": 49.00, "lon_min": -116.05, "lon_max": -104.05,
     "tribal": ["Crow", "Northern Cheyenne", "Blackfeet", "Eastern Shoshone",
          "Northern Arapaho", "Shoshone-Bannock"]},

    # --- SOUTHWEST ---
    {"name": "Navajo Nation / Four Corners", "macro": "Southwest",
     "lat_min": 35.00, "lat_max": 37.30, "lon_min": -111.85, "lon_max": -107.85,
     "tribal": ["Navajo Nation (Diné)", "Hopi", "Zuni", "Ute Mountain Ute"]},
    {"name": "Phoenix / Valley of the Sun", "macro": "Southwest",
     "lat_min": 33.05, "lat_max": 33.95, "lon_min": -112.85, "lon_max": -111.55,
     "tribal": ["Akimel O'odham", "Tohono O'odham", "Salt River Pima-Maricopa",
          "Gila River"]},
    {"name": "Tucson / Southern Arizona Borderlands", "macro": "Southwest",
     "lat_min": 31.30, "lat_max": 32.85, "lon_min": -111.55, "lon_max": -109.05,
     "tribal": ["Tohono O'odham", "Pascua Yaqui"]},
    {"name": "New Mexico", "macro": "Southwest",
     "lat_min": 31.30, "lat_max": 37.00, "lon_min": -109.05, "lon_max": -103.00,
     "tribal": ["19 Pueblos", "Mescalero Apache", "Jicarilla Apache",
          "Navajo (NW NM)"]},
    {"name": "Tulsa / Green Country / Cherokee Nation", "macro": "Southwest",
     "lat_min": 35.30, "lat_max": 37.00, "lon_min": -96.40, "lon_max": -94.45,
     "tribal": ["Cherokee Nation", "Muscogee (Creek) Nation", "Osage Nation"]},
    {"name": "Choctaw Country / Southeastern Oklahoma", "macro": "Southwest",
     "lat_min": 33.65, "lat_max": 35.30, "lon_min": -96.55, "lon_max": -94.45,
     "tribal": ["Choctaw Nation", "Chickasaw Nation"]},
    {"name": "OKC / Frontier Country", "macro": "Southwest",
     "lat_min": 34.30, "lat_max": 36.45, "lon_min": -98.55, "lon_max": -96.40,
     "tribal": ["Otoe-Missouria", "Pawnee", "Iowa", "Comanche", "Kiowa"]},
    {"name": "Texas Panhandle", "macro": "Southwest",
     "lat_min": 34.30, "lat_max": 36.50, "lon_min": -103.05, "lon_max": -100.00,
     "tribal": ["Comanche (ancestral)", "Kiowa (ancestral)"]},
    {"name": "West Texas / Permian Basin", "macro": "Southwest",
     "lat_min": 29.00, "lat_max": 36.50, "lon_min": -106.65, "lon_max": -101.10,
     "tribal": ["Mescalero Apache (ancestral)"]},
    {"name": "South Texas / Rio Grande Valley", "macro": "Southwest",
     "lat_min": 25.85, "lat_max": 28.45, "lon_min": -100.10, "lon_max": -96.85,
     "tribal": ["Coahuiltecan (ancestral)"]},
    {"name": "Texas Hill Country", "macro": "Southwest",
     "lat_min": 29.80, "lat_max": 31.20, "lon_min": -100.10, "lon_max": -97.65,
     "tribal": []},
    {"name": "East Texas / Piney Woods", "macro": "Southwest",
     "lat_min": 30.50, "lat_max": 33.95, "lon_min": -95.65, "lon_max": -93.50,
     "tribal": ["Caddo (ancestral)", "Alabama-Coushatta"]},
    {"name": "Texas Triangle", "macro": "Southwest",
     "lat_min": 29.10, "lat_max": 33.85, "lon_min": -98.95, "lon_max": -94.65,
     "tribal": []},

    # --- SOUTH ---
    {"name": "Florida Keys / Conch Republic", "macro": "South",
     "lat_min": 24.40, "lat_max": 25.30, "lon_min": -82.20, "lon_max": -80.20,
     "tribal": []},
    {"name": "South Florida (Gold Coast)", "macro": "South",
     "lat_min": 25.10, "lat_max": 28.65, "lon_min": -80.95, "lon_max": -79.85,
     "tribal": ["Seminole Tribe of Florida", "Miccosukee"]},
    {"name": "Tampa Bay / Suncoast", "macro": "South",
     "lat_min": 27.30, "lat_max": 28.45, "lon_min": -83.10, "lon_max": -82.05,
     "tribal": []},
    {"name": "Central Florida", "macro": "South",
     "lat_min": 28.10, "lat_max": 29.30, "lon_min": -81.85, "lon_max": -80.45,
     "tribal": ["Seminole (historical)"]},
    {"name": "North Florida", "macro": "South",
     "lat_min": 29.30, "lat_max": 31.00, "lon_min": -87.65, "lon_max": -81.40,
     "tribal": ["Seminole (historical)", "Muscogee (ancestral)"]},
    {"name": "Greater New Orleans", "macro": "South",
     "lat_min": 29.65, "lat_max": 30.50, "lon_min": -90.55, "lon_max": -89.55,
     "tribal": ["Houma (United Houma Nation)"]},
    {"name": "Acadiana / Cajun Country", "macro": "South",
     "lat_min": 29.45, "lat_max": 31.45, "lon_min": -94.05, "lon_max": -90.85,
     "tribal": ["Chitimacha", "Houma", "Tunica-Biloxi", "Coushatta"]},
    {"name": "North Louisiana / Ark-La-Tex", "macro": "South",
     "lat_min": 31.40, "lat_max": 33.05, "lon_min": -94.05, "lon_max": -90.85,
     "tribal": []},
    {"name": "Mississippi Heartland", "macro": "South",
     "lat_min": 30.20, "lat_max": 35.00, "lon_min": -91.65, "lon_max": -88.10,
     "tribal": ["Mississippi Band of Choctaw"]},
    {"name": "Alabama Black Belt", "macro": "South",
     "lat_min": 30.20, "lat_max": 35.00, "lon_min": -88.45, "lon_max": -84.90,
     "tribal": ["Poarch Band of Creek Indians"]},
    {"name": "Greater Atlanta", "macro": "South",
     "lat_min": 33.20, "lat_max": 34.30, "lon_min": -85.05, "lon_max": -83.50,
     "tribal": ["Muscogee (Creek) (ancestral)", "Cherokee (ancestral)"]},
    {"name": "Georgia Black Belt + Coastal GA", "macro": "South",
     "lat_min": 30.35, "lat_max": 33.90, "lon_min": -85.60, "lon_max": -80.75,
     "tribal": ["Gullah Geechee (cultural)"]},
    {"name": "West Tennessee / Mid-South / Mississippi Delta", "macro": "South",
     "lat_min": 33.00, "lat_max": 36.85, "lon_min": -91.65, "lon_max": -88.85,
     "tribal": ["Chickasaw (ancestral)"]},
    {"name": "Middle Tennessee / Nashville", "macro": "South",
     "lat_min": 35.40, "lat_max": 36.65, "lon_min": -87.50, "lon_max": -85.50,
     "tribal": []},
    {"name": "East Tennessee / Knoxville / Tri-Cities", "macro": "South",
     "lat_min": 35.00, "lat_max": 36.65, "lon_min": -85.40, "lon_max": -81.65,
     "tribal": ["Cherokee (ancestral)"]},
    {"name": "Blue Ridge / Smokies", "macro": "South",
     "lat_min": 35.10, "lat_max": 36.80, "lon_min": -84.30, "lon_max": -81.50,
     "tribal": ["Eastern Band of Cherokee Indians"]},
    {"name": "Carolina Lowcountry & Outer Banks", "macro": "South",
     "lat_min": 31.75, "lat_max": 36.60, "lon_min": -81.70, "lon_max": -75.40,
     "tribal": ["Gullah Geechee (cultural)", "Lumbee", "Catawba"]},
    {"name": "Carolina Piedmont", "macro": "South",
     "lat_min": 33.70, "lat_max": 36.55, "lon_min": -82.40, "lon_max": -78.10,
     "tribal": ["Catawba"]},
    {"name": "Appalachia (SW VA / WV / E. KY)", "macro": "South",
     "lat_min": 35.10, "lat_max": 39.95, "lon_min": -84.50, "lon_max": -78.20,
     "tribal": ["Cherokee (ancestral)", "Shawnee (ancestral)"]},

    # --- MID-ATLANTIC ---
    {"name": "Shenandoah Valley", "macro": "Mid-Atlantic",
     "lat_min": 37.30, "lat_max": 39.45, "lon_min": -79.50, "lon_max": -77.85,
     "tribal": []},
    {"name": "Richmond / Central Virginia", "macro": "Mid-Atlantic",
     "lat_min": 37.10, "lat_max": 38.45, "lon_min": -78.95, "lon_max": -76.85,
     "tribal": ["Pamunkey", "Mattaponi"]},
    {"name": "Hampton Roads / Tidewater", "macro": "Mid-Atlantic",
     "lat_min": 36.55, "lat_max": 37.40, "lon_min": -77.05, "lon_max": -75.80,
     "tribal": ["Nansemond"]},
    {"name": "Eastern Shore / Delmarva", "macro": "Mid-Atlantic",
     "lat_min": 37.95, "lat_max": 39.85, "lon_min": -76.20, "lon_max": -74.85,
     "tribal": ["Nanticoke (ancestral)"]},
    {"name": "Baltimore / Charm City", "macro": "Mid-Atlantic",
     "lat_min": 39.10, "lat_max": 39.75, "lon_min": -77.10, "lon_max": -76.20,
     "tribal": []},
    {"name": "DMV (DC, MD suburbs, NoVA)", "macro": "Mid-Atlantic",
     "lat_min": 38.40, "lat_max": 39.40, "lon_min": -77.65, "lon_max": -76.55,
     "tribal": ["Piscataway (ancestral)"]},

    # --- NORTHEAST ---
    {"name": "Western Pennsylvania / Pittsburgh", "macro": "Northeast",
     "lat_min": 39.70, "lat_max": 42.30, "lon_min": -80.55, "lon_max": -77.35,
     "tribal": ["Seneca (ancestral)", "Shawnee (ancestral)"]},
    {"name": "Pennsylvania Dutch Country / Central PA", "macro": "Northeast",
     "lat_min": 39.70, "lat_max": 41.95, "lon_min": -78.50, "lon_max": -75.30,
     "tribal": ["Lenape (ancestral)"]},
    {"name": "Greater Philadelphia / Delaware Valley", "macro": "Northeast",
     "lat_min": 38.90, "lat_max": 40.60, "lon_min": -75.60, "lon_max": -74.05,
     "tribal": ["Lenape (ancestral)"]},
    {"name": "Greater Boston", "macro": "Northeast",
     "lat_min": 41.50, "lat_max": 43.10, "lon_min": -71.90, "lon_max": -70.50,
     "tribal": ["Wampanoag", "Massachusett (ancestral)"]},
    {"name": "Northern New England", "macro": "Northeast",
     "lat_min": 43.10, "lat_max": 47.50, "lon_min": -73.45, "lon_max": -66.95,
     "tribal": ["Penobscot", "Passamaquoddy", "Abenaki", "Maliseet"]},
    {"name": "Southern New England (CT/RI)", "macro": "Northeast",
     "lat_min": 41.10, "lat_max": 42.05, "lon_min": -73.75, "lon_max": -71.10,
     "tribal": ["Mashantucket Pequot", "Mohegan", "Narragansett"]},
    {"name": "North Country / Adirondacks", "macro": "Northeast",
     "lat_min": 43.40, "lat_max": 45.05, "lon_min": -76.15, "lon_max": -73.20,
     "tribal": ["St. Regis Mohawk", "Akwesasne"]},
    {"name": "Capital Region NY", "macro": "Northeast",
     "lat_min": 42.40, "lat_max": 43.50, "lon_min": -75.50, "lon_max": -73.20,
     "tribal": ["Mohawk (ancestral)"]},
    {"name": "Hudson Valley & Catskills", "macro": "Northeast",
     "lat_min": 41.40, "lat_max": 42.40, "lon_min": -75.10, "lon_max": -73.40,
     "tribal": ["Lenape (ancestral)", "Munsee (ancestral)"]},
    {"name": "Western New York / Finger Lakes", "macro": "Northeast",
     "lat_min": 41.95, "lat_max": 43.40, "lon_min": -79.80, "lon_max": -75.60,
     "tribal": ["Haudenosaunee (Iroquois Confederacy)", "Seneca", "Cayuga",
          "Onondaga", "Oneida", "Mohawk", "Tuscarora"]},
    {"name": "Long Island", "macro": "Northeast",
     "lat_min": 40.55, "lat_max": 41.20, "lon_min": -73.80, "lon_max": -71.85,
     "tribal": ["Shinnecock", "Unkechaug"]},
    {"name": "Greater NYC / Tri-State", "macro": "Northeast",
     "lat_min": 40.45, "lat_max": 41.40, "lon_min": -74.45, "lon_max": -71.85,
     "tribal": ["Lenape (ancestral)"]},

    # --- MIDWEST ---
    {"name": "Black Hills / Lakota Country", "macro": "Midwest",
     "lat_min": 43.00, "lat_max": 45.95, "lon_min": -105.05, "lon_max": -101.90,
     "tribal": ["Oglala Lakota", "Sicangu Lakota", "Cheyenne River Sioux"]},
    {"name": "Cornbelt / Heartland Plains", "macro": "Midwest",
     "lat_min": 37.00, "lat_max": 49.00, "lon_min": -104.05, "lon_max": -90.15,
     "tribal": ["Pawnee (ancestral)", "Omaha", "Ponca", "Otoe-Missouria"]},
    {"name": "Kansas City / Heart of America", "macro": "Midwest",
     "lat_min": 38.75, "lat_max": 39.45, "lon_min": -95.05, "lon_max": -94.20,
     "tribal": []},
    {"name": "Greater St. Louis", "macro": "Midwest",
     "lat_min": 38.30, "lat_max": 39.05, "lon_min": -90.85, "lon_max": -89.85,
     "tribal": ["Osage (ancestral)"]},
    {"name": "Wisconsin / Greater Milwaukee", "macro": "Midwest",
     "lat_min": 42.50, "lat_max": 47.10, "lon_min": -92.90, "lon_max": -86.80,
     "tribal": ["Menominee", "Ho-Chunk", "Oneida", "Stockbridge-Munsee",
          "Lac du Flambeau Ojibwe"]},
    {"name": "Twin Cities + Minnesota", "macro": "Midwest",
     "lat_min": 43.50, "lat_max": 49.40, "lon_min": -97.25, "lon_max": -89.50,
     "tribal": ["Ojibwe (Anishinaabe)", "Dakota Sioux"]},
    {"name": "Indianapolis / Indiana", "macro": "Midwest",
     "lat_min": 37.75, "lat_max": 41.75, "lon_min": -88.10, "lon_max": -84.80,
     "tribal": ["Miami (ancestral)", "Pokagon Band of Potawatomi"]},
    {"name": "Columbus / Central Ohio", "macro": "Midwest",
     "lat_min": 39.55, "lat_max": 40.50, "lon_min": -83.40, "lon_max": -82.50,
     "tribal": []},
    {"name": "Cincinnati / Southwest Ohio + N. KY", "macro": "Midwest",
     "lat_min": 38.95, "lat_max": 40.10, "lon_min": -85.05, "lon_max": -83.65,
     "tribal": ["Shawnee (ancestral)", "Miami (ancestral)"]},
    {"name": "Greater Cleveland / Northeast Ohio", "macro": "Midwest",
     "lat_min": 40.85, "lat_max": 42.05, "lon_min": -82.55, "lon_max": -80.40,
     "tribal": []},
    {"name": "Northern Michigan + Upper Peninsula", "macro": "Midwest",
     "lat_min": 44.00, "lat_max": 47.50, "lon_min": -90.40, "lon_max": -83.45,
     "tribal": ["Ojibwe", "Ottawa", "Potawatomi", "Sault Ste. Marie Tribe"]},
    {"name": "West Michigan / Grand Rapids", "macro": "Midwest",
     "lat_min": 42.10, "lat_max": 44.45, "lon_min": -86.65, "lon_max": -84.70,
     "tribal": ["Ottawa (Odawa)"]},
    {"name": "Detroit / Southeast Michigan", "macro": "Midwest",
     "lat_min": 41.95, "lat_max": 43.05, "lon_min": -84.35, "lon_max": -82.40,
     "tribal": []},
    {"name": "Chicagoland", "macro": "Midwest",
     "lat_min": 41.30, "lat_max": 42.55, "lon_min": -88.50, "lon_max": -86.95,
     "tribal": ["Potawatomi (ancestral)"]},
]

# Approximate bounding boxes for all 50 US states + DC + territories.
# Format: (lat_min, lat_max, lon_min, lon_max, state_code)
# Order matters: smaller/more specific bboxes FIRST so they match
# before larger ones overlap them. Tested for non-overlap across
# major metros — borderline points (within 0.2 degrees of multiple
# state lines) get the FIRST matching bbox.
STATE_BBOXES = [
    # Territories first (most specific lat/lon)
    (-14.55, -14.10, -171.10, -169.40, "AS"), # American Samoa
    (13.20, 13.70, 144.60, 145.00, "GU"),    # Guam
    (14.10, 20.55, 144.85, 146.10, "MP"),    # Northern Mariana Islands
    (17.65, 18.45, -65.10, -64.55, "VI"),    # US Virgin Islands
    (17.85, 18.55, -67.30, -65.20, "PR"),    # Puerto Rico
    # Hawaii — distinct island chain
    (18.90, 22.30, -160.30, -154.80, "HI"),
    # Alaska — wide range
    (51.00, 71.50, -180.00, -130.00, "AK"),
    # DC — small, must come BEFORE MD and VA
    (38.79, 39.00, -77.12, -76.91, "DC"),
    # New England — small states FIRST so they don't get swallowed by NY/NH
    (41.30, 42.05, -71.90, -71.10, "RI"), # Rhode Island
    (41.00, 42.05, -73.75, -71.79, "CT"), # Connecticut
    (42.00, 42.90, -73.50, -69.90, "MA"), # Massachusetts
    (42.70, 45.00, -72.55, -70.55, "NH"), # New Hampshire
    (42.73, 45.02, -73.43, -71.46, "VT"), # Vermont
    (43.05, 47.45, -71.10, -66.95, "ME"), # Maine
    # Mid-Atlantic
    (38.45, 39.72, -75.79, -74.98, "DE"),
    (37.90, 39.73, -79.49, -75.04, "MD"),
    (38.90, 41.36, -75.56, -73.90, "NJ"),
    (39.72, 42.27, -80.52, -74.69, "PA"),
    (40.50, 45.02, -79.76, -71.85, "NY"),
    (36.54, 39.47, -83.68, -75.24, "VA"),
    (37.20, 40.64, -82.64, -77.72, "WV"),
    # South
    (33.78, 36.59, -84.32, -75.46, "NC"),
    (32.03, 35.22, -83.35, -78.54, "SC"),
    (30.36, 35.00, -85.61, -80.84, "GA"),
    (24.40, 31.00, -87.63, -79.97, "FL"),
    (30.20, 35.01, -88.47, -84.89, "AL"),
    (35.00, 36.68, -90.31, -81.65, "TN"),
    (30.17, 35.00, -91.65, -88.10, "MS"),
    (28.93, 33.02, -94.05, -88.76, "LA"),
    (33.00, 36.50, -94.62, -89.64, "AR"),
# Midwest — IL, IN, OH BEFORE MI/WI to prevent western-overlap claims
    (35.00, 40.00, -103.00, -94.43, "OK"),  # OK before TX/KS
    (25.84, 36.50, -106.65, -93.51, "TX"),
    (36.99, 40.00, -102.05, -94.59, "KS"),
    (40.00, 43.00, -104.05, -95.30, "NE"),
    (37.00, 40.62, -95.77, -89.10, "MO"),
    (40.38, 43.50, -96.64, -90.14, "IA"),
    (45.94, 49.00, -104.05, -96.55, "ND"),
    (42.48, 45.94, -104.06, -96.44, "SD"),
    (42.49, 49.40, -97.24, -89.50, "MN"),
    # Smaller Great Lakes states FIRST so they match before MI/WI westward bbox swallows them
    (37.77, 42.51, -91.51, -87.50, "IL"),
    (37.77, 41.76, -88.10, -84.78, "IN"),
    (38.40, 42.00, -84.82, -80.52, "OH"),
    (38.40, 39.15, -89.57, -81.96, "KY"),  # KY moved here too — was after South, but its lat overlaps OH
    # Now the larger Great Lakes states with corrected lon_min
    (42.49, 47.08, -92.89, -86.81, "WI"),  # tightened lon_max from -86.25 to -86.81 (Lake Michigan east shore)
    (41.70, 48.31, -90.42, -82.41, "MI"),
    # Mountain West
    (41.00, 49.00, -116.05, -104.04, "MT"),
    (44.00, 49.00, -117.24, -111.05, "ID"),
    (40.99, 45.01, -111.06, -104.05, "WY"),
    (36.99, 41.00, -109.06, -102.04, "CO"),
    (36.99, 42.00, -114.06, -109.04, "UT"),
    (35.00, 42.00, -120.01, -114.04, "NV"),
    (31.33, 37.00, -114.82, -109.04, "AZ"),
    (31.33, 37.00, -109.05, -103.00, "NM"),
    # Pacific
    (45.54, 49.00, -124.84, -116.91, "WA"),
    (41.99, 46.30, -124.57, -116.46, "OR"),
    (32.53, 42.00, -124.49, -114.13, "CA"),
]

def state_from_latlon(lat: float, lon: float) -> str:
    """Returns 2-letter US state/territory code for a given lat/lon.
    Returns 'XX' as a last-resort fallback only when coordinates are
    outside all known US bounding boxes."""
    for lat_min, lat_max, lon_min, lon_max, code in STATE_BBOXES:
        if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
            return code
    return "XX"


class SportInHub(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sport: str
    count: int
    paralympic_count: int
    track_type: str


class HubComposition(BaseModel):
    model_config = ConfigDict(extra="forbid")
    olympic_count: int
    paralympic_count: int
    both_count: int
    paralympic_share: float
    composition_label: str


class Hub(BaseModel):
    model_config = ConfigDict(extra="forbid")
    hub_id: str
    display_name: str
    centroid_latitude: float
    centroid_longitude: float
    medoid_hometown: str
    radius_km: float
    region: str
    region_name: str
    macro_region: str
    states: list[str]
    total_athletes: int
    composition: HubComposition
    is_paralympic_hot_spot: bool
    top_sports: list[SportInHub]
    sport_diversity_index: float
    tags: list[str]
    search_aliases: list[str]


class ClusteredAthlete(BaseModel):
    model_config = ConfigDict(extra="forbid")
    wikidata_uri: str
    name: str
    status: OlympicParalympicStatus
    sports: list[str]
    hometown: Hometown
    medals: list[str]
    gender: str | None = None
    date_of_birth: str | None = None
    hub_id: str
    is_core_member: bool
    distance_to_hub_km: float


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return EARTH_RADIUS_KM * c


def get_state_from_city(city_label: str) -> str:
    for city, state in CITY_TO_STATE.items():
        if city.lower() in city_label.lower():
            return state
    return "XX"


def get_region(lat: float, lon: float) -> str:
    if lat >= 38 and lon >= -80:
        return "northeast"
    if lat >= 36 and -100 <= lon < -80:
        return "midwest"
    if lat < 36 and lon >= -100:
        return "south"
    if lat >= 31 and -115 <= lon < -100:
        return "mountain"
    if lat >= 31 and lon < -115:
        return "pacific"
    return "west"


def classify_region(lat: float, lon: float) -> tuple[str, str]:
    """Returns (region_name, macro_region) for a given lat/lon.
    Returns ('Continental US', 'Other') if no match (rare fallback)."""
    for region in REGIONAL_CONTEXTS:
        if (region["lat_min"] <= lat <= region["lat_max"]
            and region["lon_min"] <= lon <= region["lon_max"]):
            return (region["name"], region["macro"])
    return ("Continental US", "Other")


def main() -> None:
    base_dir = Path("pipeline")
    in_path = base_dir / "geocoded" / "athletes.json"
    athletes_out_path = base_dir / "clustered" / "athletes.json"
    hubs_out_path = base_dir / "clustered" / "hubs.json"

    if not in_path.exists():
        logger.error(f"Input file not found: {in_path}")
        return

    logger.info("Loading geocoded athletes...")
    with in_path.open("r", encoding="utf-8") as f:
        raw_data = json.load(f)

    athletes: list[Athlete] = []
    for item in raw_data:
        athletes.append(Athlete.model_validate(item))

    valid_athletes = []
    dropped_athletes = []
    for a in athletes:
        if a.hometown.latitude is not None and a.hometown.longitude is not None:
            valid_athletes.append(a)
        else:
            dropped_athletes.append(a)

    logger.info(f"Dropped {len(dropped_athletes)} athletes due to missing coordinates.")

    if not valid_athletes:
        logger.error("No athletes with valid coordinates to cluster.")
        return

    coords = np.array([[a.hometown.latitude, a.hometown.longitude] for a in valid_athletes])
    coords_rad = np.radians(coords)

    logger.info("Running HDBSCAN Stage 1...")
    clusterer = HDBSCAN(
        min_cluster_size=20,
        min_samples=8,
        cluster_selection_epsilon=EPSILON_RADIANS,
        metric="haversine",
        cluster_selection_method="eom",
        store_centers="medoid",
        allow_single_cluster=False,
    )
    clusterer.fit(coords_rad)

    labels = clusterer.labels_.copy()
    raw_cluster_count = len(set(labels) - {-1})
    logger.info(f"Discovered {raw_cluster_count} raw clusters. Stage 1 complete.")

    if clusterer.medoids_ is None or raw_cluster_count == 0:
        logger.error("HDBSCAN failed to find any clusters.")
        return

    medoids_deg = np.degrees(clusterer.medoids_)

    logger.info("Running Stage 2: Nearest-medoid assignment for noise points...")
    is_core_member = (labels != -1)
    
    for i, label in enumerate(labels):
        if label == -1:
            lat1, lon1 = coords[i]
            min_dist = float('inf')
            closest_c = -1
            for c_idx, (m_lat, m_lon) in enumerate(medoids_deg):
                dist = haversine_km(lat1, lon1, m_lat, m_lon)
                if dist < min_dist:
                    min_dist = dist
                    closest_c = c_idx
            labels[i] = closest_c

    cluster_to_athletes: dict[int, list[tuple[Athlete, bool, float, float]]] = {}
    for c_idx in range(len(medoids_deg)):
        cluster_to_athletes[c_idx] = []

    for i, c_idx in enumerate(labels):
        cluster_to_athletes[c_idx].append((valid_athletes[i], bool(is_core_member[i]), coords[i][0], coords[i][1]))

    logger.info("Running Stage 3 & 4: Hub generation and metadata computation...")
    hubs: list[Hub] = []
    clustered_athletes_out: list[ClusteredAthlete] = []

    winter_sports = {
        "figure skating", "ice hockey", "alpine skiing", "speed skating", 
        "snowboarding", "cross-country skiing", "bobsleigh", "luge", 
        "skeleton", "curling", "biathlon", "freestyle skiing", 
        "nordic combined", "ski jumping", "short track speed skating"
    }

    for c_idx, (m_lat, m_lon) in enumerate(medoids_deg):
        members = cluster_to_athletes[c_idx]
        if not members:
            continue

        min_m_dist = float('inf')
        medoid_hometown = ""
        for a, core, a_lat, a_lon in members:
            dist = haversine_km(m_lat, m_lon, a_lat, a_lon)
            if dist < min_m_dist:
                min_m_dist = dist
                medoid_hometown = a.hometown.label

        city_slug = re.sub(r'[^A-Z0-9_]', '', medoid_hometown.upper().replace(' ', '_'))
        state = state_from_latlon(m_lat, m_lon)
        hub_id = f"HUB_{state}_{city_slug}"

        if medoid_hometown in DISPLAY_NAME_OVERRIDES:
            display_name = DISPLAY_NAME_OVERRIDES[medoid_hometown]
        else:
            display_name = f"{medoid_hometown} Region, {state}"

        distances = []
        for a, core, a_lat, a_lon in members:
            distances.append(haversine_km(m_lat, m_lon, a_lat, a_lon))
        radius_km = float(np.percentile(distances, 95)) if distances else 0.0

        region = get_region(m_lat, m_lon)
        region_name, macro_region = classify_region(m_lat, m_lon)

        state_counter: Counter[str] = Counter()
        for a, _, a_lat, a_lon in members:
            state_counter[state_from_latlon(a_lat, a_lon)] += 1
        top_states = [s for s, _ in state_counter.most_common(3)]

        oly_c = sum(1 for a, _, _, _ in members if a.status == OlympicParalympicStatus.OLYMPIC)
        para_c = sum(1 for a, _, _, _ in members if a.status == OlympicParalympicStatus.PARALYMPIC)
        both_c = sum(1 for a, _, _, _ in members if a.status == OlympicParalympicStatus.BOTH)
        total = len(members)

        para_share = (para_c + both_c) / total if total > 0 else 0.0

        if para_share >= 0.15:
            comp_label = "paralympic_strong"
        elif para_share >= 0.05:
            comp_label = "balanced"
        else:
            comp_label = "olympic_dominant"

        composition = HubComposition(
            olympic_count=oly_c,
            paralympic_count=para_c,
            both_count=both_c,
            paralympic_share=para_share,
            composition_label=comp_label
        )

        is_hot_spot = para_share >= 0.092

        sport_counter: Counter[str] = Counter()
        sport_para_counter: Counter[str] = Counter()
        for a, _, _, _ in members:
            is_para_athlete = a.status in (OlympicParalympicStatus.PARALYMPIC, OlympicParalympicStatus.BOTH)
            for s in a.sports:
                sport_counter[s] += 1
                if is_para_athlete:
                    sport_para_counter[s] += 1

        top_sports = []
        for s, count in sport_counter.most_common(3):
            p_count = sport_para_counter[s]
            if p_count == 0:
                t_type = "olympic"
            elif p_count == count:
                t_type = "paralympic"
            else:
                t_type = "both"
            top_sports.append(SportInHub(sport=s, count=count, paralympic_count=p_count, track_type=t_type))

        H = 0.0
        total_sports_count = sum(sport_counter.values())
        if total_sports_count > 0:
            for count in sport_counter.values():
                p = count / total_sports_count
                H -= p * math.log(p)
        num_unique = len(sport_counter)
        H_max = math.log(num_unique) if num_unique > 0 else 0.0
        diversity = (H / H_max) if H_max > 0 else 0.0

        tags = []
        if is_hot_spot:
            tags.append("para-hot-spot")
        has_winter = any(ts.sport.lower() in winter_sports for ts in top_sports)
        if has_winter:
            tags.append("winter-strong")
        else:
            tags.append("summer-strong")
        tags.append(region)
        if top_sports:
            sport_slug = re.sub(r'[^a-z0-9]+', '-', top_sports[0].sport.lower()).strip('-')
            tags.append(sport_slug)

        search_aliases = []
        label_lower = medoid_hometown.lower()
        if "new york" in label_lower:
            search_aliases.extend(["NYC", "Manhattan", "NY"])
        elif "minneapolis" in label_lower:
            search_aliases.extend(["Twin Cities", "MSP"])
        elif "san francisco" in label_lower:
            search_aliases.extend(["SF", "Bay Area"])
        elif "los angeles" in label_lower:
            search_aliases.extend(["LA", "L.A."])

        search_aliases.insert(0, medoid_hometown)
        if state not in search_aliases:
            search_aliases.append(state)

        hub = Hub(
            hub_id=hub_id,
            display_name=display_name,
            centroid_latitude=m_lat,
            centroid_longitude=m_lon,
            medoid_hometown=medoid_hometown,
            radius_km=radius_km,
            region=region,
            region_name=region_name,
            macro_region=macro_region,
            states=top_states,
            total_athletes=total,
            composition=composition,
            is_paralympic_hot_spot=is_hot_spot,
            top_sports=top_sports,
            sport_diversity_index=diversity,
            tags=tags,
            search_aliases=search_aliases
        )
        hubs.append(hub)

        for a, core, a_lat, a_lon in members:
            dist = haversine_km(m_lat, m_lon, a_lat, a_lon)
            ca = ClusteredAthlete(
                wikidata_uri=a.wikidata_uri,
                name=a.name,
                status=a.status,
                sports=a.sports,
                hometown=a.hometown,
                medals=a.medals,
                gender=a.gender,
                date_of_birth=a.date_of_birth,
                hub_id=hub.hub_id,
                is_core_member=core,
                distance_to_hub_km=dist
            )
            clustered_athletes_out.append(ca)

    hubs.sort(key=lambda x: x.total_athletes, reverse=True)

    athletes_out_path.parent.mkdir(parents=True, exist_ok=True)
    
    with athletes_out_path.open("w", encoding="utf-8") as f:
        json.dump([a.model_dump(mode="json") for a in clustered_athletes_out], f, indent=2, ensure_ascii=False)
        
    with hubs_out_path.open("w", encoding="utf-8") as f:
        json.dump([h.model_dump(mode="json") for h in hubs], f, indent=2, ensure_ascii=False)

    logger.info("--- CLUSTERING SUMMARY ---")
    logger.info(f"Total clusters discovered: {raw_cluster_count}")
    core_count = sum(1 for a in clustered_athletes_out if a.is_core_member)
    noise_count = len(clustered_athletes_out) - core_count
    logger.info(f"Total athletes assigned: {len(clustered_athletes_out)} ({core_count} core, {noise_count} nearest-medoid)")
    
    logger.info("Hubs with composition_label = 'paralympic_strong':")
    for h in hubs:
        if h.composition.composition_label == "paralympic_strong":
            logger.info(f"  {h.hub_id} ({h.display_name}): {h.composition.paralympic_share:.1%} para share")
            
    logger.info("Hubs flagged is_paralympic_hot_spot:")
    for h in hubs:
        if h.is_paralympic_hot_spot:
            logger.info(f"  {h.hub_id} ({h.display_name}): {h.composition.paralympic_share:.1%} para share")

    logger.info("Top 5 hubs by total_athletes:")
    for h in hubs[:5]:
        logger.info(f"  {h.hub_id} ({h.display_name}) [{h.region_name}, {h.macro_region}]: {h.total_athletes} athletes")

    logger.info(f"Athletes dropped due to missing coordinates: {len(dropped_athletes)}")


if __name__ == "__main__":
    main()