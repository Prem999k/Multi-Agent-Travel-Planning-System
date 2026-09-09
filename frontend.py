"""
TravelMind AI — Streamlit frontend.

This module is presentation-only. All planning logic (LangGraph, MCP
servers, PostgreSQL checkpointing) lives in main.py / mcp_client.py and is
untouched here — this file adapts to that backend's existing output shape
rather than the other way around.

The compiled graph currently ends at `itinerary_agent` (there is no
`final_agent` node); that node's output is treated as the final plan.
"""

import ast
import io
import os
import json
import logging
import re
import uuid
from collections import OrderedDict
from datetime import date, datetime, timedelta

import streamlit as st
import streamlit.components.v1 as components
from langchain_core.messages import HumanMessage
from langgraph.types import Command

from graph import app  # compiled LangGraph app with PostgreSQL checkpointing

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("travelmind.frontend")

# ─────────────────────────────────────────────────────────────────────────
# Brand Identity & Constants
# ─────────────────────────────────────────────────────────────────────────
APP_NAME = "TravelMind AI"
APP_TAGLINE = "Plan smarter. Travel better."
APP_SUBTITLE = "Intelligent Multi-Agent Travel Planning System"
CREATOR_CREDIT = "Built & designed by Prem Kumar ✦"

PREFERENCE_OPTIONS = [
    "Sightseeing", "Food", "Adventure", "Nature",
    "Culture", "Shopping", "Relaxation", "Nightlife",
]

EXAMPLE_PROMPTS = {
    "Tirupati": dict(
        destination="Tirupati, Andhra Pradesh",
        origin="Hyderabad, India",
        days=3,
        travelers=4,
        budget=25000,
        preferences=["Culture", "Sightseeing", "Relaxation"],
        additional="Prioritize Tirumala Sri Venkateswara Temple, darshan planning, affordable accommodation and practical local transport."
    ),
    "Kailasa Temple": dict(
        destination="Ellora, Maharashtra",
        origin="Hyderabad, India",
        days=4,
        travelers=4,
        budget=35000,
        preferences=["Culture", "Sightseeing", "Adventure"],
        additional="Prioritize the Kailasa Temple at Ellora, the giant Shiva temple carved from a single rock, nearby Ellora caves and practical low-cost travel."
    ),
    "Ram Mandir": dict(
        destination="Ayodhya, Uttar Pradesh",
        origin="Hyderabad, India",
        days=3,
        travelers=4,
        budget=30000,
        preferences=["Culture", "Sightseeing", "Relaxation"],
        additional="Prioritize Shri Ram Mandir, Hanuman Garhi, Saryu riverfront and a comfortable spiritual itinerary."
    ),
    "Kashmir + Ladakh Bike Road Trip": dict(
        destination="Kashmir + Ladakh",
        origin="Hyderabad, India",
        days=10,
        travelers=5,
        budget=150000,
        preferences=["Adventure", "Nature", "Sightseeing"],
        additional="Plan as a 5-member motorcycle road trip. Include bike rental, permits, realistic Himalayan riding times, acclimatization, fuel stops, safety, weather buffers, and peaceful Shiva/Bholenath-inspired Himalayan experiences where practical."
    ),
    "Kerala Nature": dict(
        destination="Kerala, India",
        origin="Hyderabad, India",
        days=6,
        travelers=4,
        budget=60000,
        preferences=["Nature", "Relaxation", "Food"],
        additional="Focus on Kerala nature, Munnar, backwaters, waterfalls and relaxed scenic travel."
    ),
    "Varanasi": dict(
        destination="Varanasi, Uttar Pradesh",
        origin="Hyderabad, India",
        days=3,
        travelers=4,
        budget=30000,
        preferences=["Culture", "Food", "Sightseeing"],
        additional="Prioritize Kashi Vishwanath Temple, Ganga Aarti, ghats, sunrise boat ride and the old-city food experience."
    ),
    "Rajasthan": dict(
        destination="Rajasthan, India",
        origin="Hyderabad, India",
        days=6,
        travelers=4,
        budget=70000,
        preferences=["Culture", "Sightseeing", "Food"],
        additional="Build a practical Jaipur-Jodhpur-Jaisalmer/Udaipur route with iconic forts, palaces, local cuisine and sensible intercity travel."
    ),
    "Tokyo Explorer": dict(
        destination="Tokyo, Japan",
        origin="Hyderabad, India",
        days=7,
        travelers=2,
        budget=200000,
        preferences=["Sightseeing", "Food", "Culture"],
        additional="First-time Tokyo trip with iconic sights, efficient transit and memorable food experiences."
    ),
    "New York Highlights": dict(
        destination="New York City, USA",
        origin="Hyderabad, India",
        days=7,
        travelers=2,
        budget=700000,
        preferences=["Sightseeing", "Food", "Culture"],
        additional="Prioritize iconic landmarks and practical public transport."
    ),
    "Paris Classics": dict(
        destination="Paris, France",
        origin="Hyderabad, India",
        days=5,
        travelers=2,
        budget=250000,
        preferences=["Culture", "Sightseeing", "Food"],
        additional="Iconic first-time Paris experience with efficient routing."
    ),
    "Swiss Alps": dict(
        destination="Switzerland",
        origin="Hyderabad, India",
        days=7,
        travelers=2,
        budget=350000,
        preferences=["Nature", "Adventure", "Sightseeing"],
        additional="Prioritize the Matterhorn, Swiss Alps, Lucerne, Interlaken and scenic rail travel."
    ),
    "London First Trip": dict(
        destination="London, UK",
        origin="Hyderabad, India",
        days=6,
        travelers=2,
        budget=300000,
        preferences=["Culture", "Sightseeing", "Food"],
        additional="First-time London highlights with efficient transit."
    ),
    "Dubai Getaway": dict(
        destination="Dubai, UAE",
        origin="Hyderabad, India",
        days=5,
        travelers=2,
        budget=180000,
        preferences=["Sightseeing", "Shopping", "Food"],
        additional="Balance iconic skyline sights, shopping and a desert experience."
    ),
    "Bali Escape": dict(
        destination="Bali, Indonesia",
        origin="Hyderabad, India",
        days=6,
        travelers=2,
        budget=140000,
        preferences=["Nature", "Relaxation", "Adventure"],
        additional="Beach time, temples, waterfalls and scenic day trips."
    ),
    "Singapore City": dict(
        destination="Singapore",
        origin="Hyderabad, India",
        days=5,
        travelers=2,
        budget=180000,
        preferences=["Sightseeing", "Food", "Shopping"],
        additional="Efficient city itinerary with Gardens by the Bay and Sentosa."
    ),
    "Maldives Escape": dict(
        destination="Maldives",
        origin="Hyderabad, India",
        days=5,
        travelers=2,
        budget=220000,
        preferences=["Relaxation", "Nature", "Food"],
        additional="A scenic couple-friendly island escape with beach time, snorkeling, a resort stay, sunset experiences and a relaxed pace."
    ),
    "China Highlights": dict(
        destination="China",
        origin="Hyderabad, India",
        days=7,
        travelers=4,
        budget=240000,
        preferences=["Culture", "Sightseeing", "Food"],
        additional="First-time China highlights focused on Beijing, the Great Wall, cultural landmarks, practical transit and local food."
    ),
    "Russia Explorer": dict(
        destination="Russia",
        origin="Hyderabad, India",
        days=7,
        travelers=2,
        budget=260000,
        preferences=["Culture", "Sightseeing", "Food"],
        additional="A first-time Moscow and Saint Petersburg journey with major landmarks, efficient transit and a balanced cultural itinerary."
    ),
    "Thailand Island Hop": dict(
        destination="Thailand",
        origin="Hyderabad, India",
        days=6,
        travelers=4,
        budget=140000,
        preferences=["Adventure", "Food", "Nature"],
        additional="Friends trip combining Bangkok with an island escape, street food, beaches, temples and practical transfers."
    ),
    "Greece Escape": dict(
        destination="Greece",
        origin="Hyderabad, India",
        days=7,
        travelers=2,
        budget=280000,
        preferences=["Culture", "Nature", "Relaxation"],
        additional="Couple-friendly Greece itinerary with Athens, island scenery, ancient sites, sunset viewpoints and sensible ferry planning."
    ),
}

EXAMPLE_GROUP_TYPES = {
    "Tirupati": "Family",
    "Kailasa Temple": "Family",
    "Ram Mandir": "Family",
    "Kashmir + Ladakh Bike Road Trip": "Friends",
    "Kerala Nature": "Family",
    "Varanasi": "Family",
    "Rajasthan": "Friends",
    "Tokyo Explorer": "Friends",
    "New York Highlights": "Friends",
    "Paris Classics": "Couple",
    "Swiss Alps": "Friends",
    "London First Trip": "Friends",
    "Dubai Getaway": "Friends",
    "Bali Escape": "Friends",
    "Singapore City": "Friends",
    "Maldives Escape": "Couple",
    "China Highlights": "Family",
    "Russia Explorer": "Friends",
    "Thailand Island Hop": "Friends",
    "Greece Escape": "Couple",
}

DESTINATION_CARDS = [
    # 🇮🇳 INDIA — 7 Curated Heritage & Scenic Escapes
    ("🛕", "Tirupati", "3 days · Tirumala + devotion", "Tirupati, Andhra Pradesh",
     "https://www.indiapilgrimtours.com/articles/wp-content/uploads/2018/01/tirumala-tirupati-venkateswara-temple.jpg",
     "Tirupati"),

    ("🪨", "Kailasa Temple", "4 days · Shiva + rock-cut wonder", "Ellora, Maharashtra",
     "https://curriculture.in/wp-content/uploads/2024/08/kailasa-kailash-temple-ellora-caves-1-scaled.jpg",
     "Kailasa Temple"),

    ("🚩", "Ram Mandir", "3 days · Shri Ram + Ayodhya", "Ayodhya, Uttar Pradesh",
     "https://static.wixstatic.com/media/229d48_e49353b75fa042f38aae54200304369f~mv2.png/v1/fill/w_980%2Ch_653%2Cal_c%2Cq_90%2Cusm_0.66_1.00_0.01%2Cenc_avif%2Cquality_auto/229d48_e49353b75fa042f38aae54200304369f~mv2.png",
     "Ram Mandir"),

    ("🏍️", "Kashmir + Ladakh", "10 days · 5 riders · Himalayan road trip", "Kashmir + Ladakh",
     "https://cdn.travelcoffee.in/uploads/43f09be2-3adb-4333-90b5-89b033d8907f-Best_Time_for_a_Ladakh_Bike_Trip.jpg",
     "Kashmir + Ladakh Bike Road Trip"),

    ("🌴", "Kerala", "6 days · backwaters + green escapes", "Kerala, India",
     "https://images.unsplash.com/photo-1602216056096-3b40cc0c9944?auto=format&fit=crop&w=1200&q=90",
     "Kerala Nature"),

    ("🪔", "Varanasi", "3 days · Kashi + Ganga", "Varanasi, Uttar Pradesh",
     "https://images.unsplash.com/photo-1561361058-c24cecae35ca?auto=format&fit=crop&w=1200&q=90",
     "Varanasi"),

    ("🏰", "Rajasthan", "6 days · forts + royal cities", "Rajasthan, India",
     "https://images.unsplash.com/photo-1477587458883-47145ed94245?auto=format&fit=crop&w=1200&q=90",
     "Rajasthan"),

    # 🌍 INTERNATIONAL — 13 Curated Global Destinations
    ("🇯🇵", "Tokyo", "7 days · food + neon", "Tokyo, Japan",
     "https://images.unsplash.com/photo-1540959733332-eab4deabeeaf?auto=format&fit=crop&w=1200&q=90",
     "Tokyo Explorer"),

    ("🗽", "New York", "7 days · city icons", "New York City, USA",
     "https://images.unsplash.com/photo-1485871981521-5b1fd3805eee?auto=format&fit=crop&w=1200&q=90",
     "New York Highlights"),

    ("🗼", "Paris", "5 days · art + romance", "Paris, France",
     "https://images.unsplash.com/photo-1502602898657-3e91760cbb34?auto=format&fit=crop&w=1200&q=90",
     "Paris Classics"),

    ("🇨🇭", "Switzerland", "7 days · Matterhorn + Alps", "Switzerland",
     "https://mountain-madness.imgix.net/general-images/Climbing-trekking-ski-mountain-guide-company-and-school-shutterstock_1504792100.f1698079682.jpg?auto=compress%2Cformat&crop=focalpoint&fit=crop&fp-x=0.5&fp-y=0.5&h=971&q=80&s=90695f65a7c8951c74f8b2719739ae96&w=2000",
     "Swiss Alps"),

    ("🇬🇧", "London", "6 days · royal + iconic", "London, UK",
     "https://images.unsplash.com/photo-1513635269975-59663e0ac1ad?auto=format&fit=crop&w=1200&q=90",
     "London First Trip"),

    ("🏙️", "Dubai", "5 days · skyline + desert", "Dubai, UAE",
     "https://images.unsplash.com/photo-1512453979798-5ea266f8880c?auto=format&fit=crop&w=1200&q=90",
     "Dubai Getaway"),

    ("🌺", "Bali", "6 days · beaches + temples", "Bali, Indonesia",
     "https://images.unsplash.com/photo-1537996194471-e657df975ab4?auto=format&fit=crop&w=1200&q=90",
     "Bali Escape"),

    ("🌃", "Singapore", "5 days · city + gardens", "Singapore",
     "https://images.unsplash.com/photo-1525625293386-3f8f99389edd?auto=format&fit=crop&w=1200&q=90",
     "Singapore City"),

    ("🏝️", "Maldives", "5 days · lagoons + slow luxury", "Maldives",
     "https://images.unsplash.com/photo-1514282401047-d79a71a590e8?auto=format&fit=crop&w=1200&q=90",
     "Maldives Escape"),

    ("🐉", "China", "7 days · heritage + modern cities", "China",
     "https://images.unsplash.com/photo-1508804185872-d7badad00f7d?auto=format&fit=crop&w=1200&q=90",
     "China Highlights"),

    ("🏛️", "Russia", "7 days · Moscow + Petersburg", "Russia",
     "https://images.unsplash.com/photo-1520106212299-d99c443e4568?auto=format&fit=crop&w=1200&q=90",
     "Russia Explorer"),

    ("🌊", "Thailand", "6 days · islands + street food", "Thailand",
     "https://images.unsplash.com/photo-1552465011-b4e21bf6e79a?auto=format&fit=crop&w=1200&q=90",
     "Thailand Island Hop"),

    ("🏛️", "Greece", "7 days · islands + ancient sites", "Greece",
     "https://images.unsplash.com/photo-1603565816030-6b389eeb23cb?auto=format&fit=crop&w=1200&q=90",
     "Greece Escape"),
]

PIPELINE_STEPS = [
    ("supervisor", "🧠 Supervisor", "Understanding your trip"),
    ("flight_agent", "✈️ Flight Agent", "Checking routes and travel options"),
    ("hotel_agent", "🏨 Hotel Agent", "Finding suitable areas and stays"),
    ("weather_agent", "🌦️ Weather Agent", "Checking weather and conditions"),
    ("budget_agent", "💰 Budget Agent", "Optimizing trip costs"),
    ("itinerary_agent", "🗺️ Itinerary Agent", "Building your day-by-day journey"),
    ("human_approval", "👤 Human Review", "Ready for your approval"),
]


# ─────────────────────────────────────────────────────────────────────────
# Styling — Forest Green Palette & Cinematic Aesthetics
# ─────────────────────────────────────────────────────────────────────────
def inject_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300..900;1,9..40,300..900&family=Space+Grotesk:wght@400;500;600;700&display=swap');

        :root {
            /* Forest Green Core System */
            --bg-deep: #050B08;
            --bg-primary: #07100B;
            --bg-surface: #09150F;
            --bg-card: #0B1710;
            --bg-card-hover: #0E1D15;
            --bg-card-elevated: #11251A;

            /* Forest Green Tones */
            --forest-deep: #071A12;
            --forest-mid: #0A2117;
            --forest-light: #0D2B1D;

            /* Pine Accents */
            --pine-dark: #123A28;
            --pine-mid: #164A33;
            --pine-light: #1D5A3D;

            /* Emerald Accents */
            --emerald-deep: #238B5B;
            --emerald-primary: #2BA66F;
            --emerald-bright: #35B97A;
            --emerald-glow: rgba(43, 166, 111, 0.28);

            /* Soft Mint */
            --mint-soft: #8FE3B5;
            --mint-light: #B7F3D0;

            /* Warm Travel Accents (Gold) */
            --gold-warm: #F2C879;
            --gold-accent: #E8B95D;
            --gold-glow: rgba(242, 200, 121, 0.24);

            /* Sky Accent (Subtle Cyan for Climate & Transport) */
            --sky-subtle: #8BD5E8;

            /* Borders */
            --border-subtle: rgba(143, 227, 181, 0.12);
            --border-mid: rgba(143, 227, 181, 0.20);
            --border-gold: rgba(242, 200, 121, 0.22);
            --border-active: rgba(53, 185, 122, 0.65);

            /* Typography */
            --text-primary: #F4FAF6;
            --text-secondary: #B5C8BD;
            --text-muted: #71867A;

            /* Radii */
            --radius-sm: 10px;
            --radius-md: 16px;
            --radius-lg: 22px;
            --radius-xl: 28px;
        }

        /* Global Resets */
        html, body, .stApp {
            background-color: var(--bg-deep) !important;
            background-image:
                radial-gradient(circle at 50% 0%, rgba(22, 74, 51, 0.35) 0%, transparent 60%),
                radial-gradient(circle at 90% 40%, rgba(13, 43, 29, 0.25) 0%, transparent 45%),
                linear-gradient(180deg, var(--bg-deep) 0%, var(--bg-primary) 100%) !important;
            color: var(--text-primary) !important;
            font-family: 'DM Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
            overflow-x: hidden !important;
        }

        * {
            box-sizing: border-box !important;
        }

        h1, h2, h3, h4, h5, h6 {
            font-family: 'Space Grotesk', -apple-system, BlinkMacSystemFont, sans-serif !important;
            letter-spacing: -0.025em !important;
            color: var(--text-primary) !important;
        }

        #MainMenu, footer, header {
            visibility: hidden !important;
            height: 0 !important;
        }

        .block-container {
            max-width: 1260px !important;
            padding: 1.2rem 2rem 4.5rem !important;
            overflow-x: hidden !important;
        }

        /* Sidebar */
        [data-testid="stSidebar"] {
            background: #07100B !important;
            border-right: 1px solid var(--border-subtle) !important;
        }
        [data-testid="stSidebar"] > div:first-child {
            padding-top: 1.4rem !important;
            padding-left: 1.2rem !important;
            padding-right: 1.2rem !important;
        }

        /* Form Inputs */
        .stTextInput input, .stTextArea textarea, .stNumberInput input, .stDateInput input {
            background: #09150F !important;
            color: var(--text-primary) !important;
            border: 1px solid var(--border-subtle) !important;
            border-radius: var(--radius-md) !important;
            padding: 0.75rem 1rem !important;
            font-size: 0.92rem !important;
            transition: all 0.22s ease !important;
        }
        .stTextInput input:focus, .stTextArea textarea:focus, .stNumberInput input:focus, .stDateInput input:focus {
            border-color: var(--emerald-primary) !important;
            box-shadow: 0 0 0 3px rgba(43, 166, 111, 0.18) !important;
            background: #0D2B1D !important;
        }
        .stTextInput label, .stTextArea label, .stNumberInput label, .stDateInput label {
            color: var(--text-secondary) !important;
            font-size: 0.84rem !important;
            font-weight: 600 !important;
            letter-spacing: 0.01em !important;
        }
        .stMultiSelect [data-baseweb="select"], .stSelectbox [data-baseweb="select"] {
            background: #09150F !important;
            border: 1px solid var(--border-subtle) !important;
            border-radius: var(--radius-md) !important;
        }
        .stMultiSelect [data-baseweb="tag"] {
            background: rgba(43, 166, 111, 0.16) !important;
            border: 1px solid rgba(143, 227, 181, 0.3) !important;
            border-radius: 999px !important;
        }
        .stMultiSelect [data-baseweb="tag"] span {
            color: var(--mint-soft) !important;
            font-weight: 600 !important;
        }

        /* Primary Call-To-Action Button */
        div[data-testid="stFormSubmitButton"] > button, .tm-primary button {
            background: linear-gradient(135deg, #35B97A 0%, #238B5B 100%) !important;
            color: #050B08 !important;
            border: none !important;
            border-radius: 14px !important;
            font-weight: 800 !important;
            font-size: 1.05rem !important;
            letter-spacing: -0.01em !important;
            min-height: 52px !important;
            transition: all 0.24s cubic-bezier(0.16, 1, 0.3, 1) !important;
            box-shadow: 0 8px 26px rgba(43, 166, 111, 0.28) !important;
            width: 100% !important;
        }
        div[data-testid="stFormSubmitButton"] > button:hover, .tm-primary button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 12px 34px rgba(43, 166, 111, 0.42) !important;
            filter: brightness(1.06) !important;
        }
        div[data-testid="stFormSubmitButton"] > button:active, .tm-primary button:active {
            transform: translateY(0) !important;
        }

        /* Standard Buttons */
        .stButton > button {
            border-radius: var(--radius-sm) !important;
            border: 1px solid var(--border-subtle) !important;
            background: #0B1710 !important;
            color: var(--text-primary) !important;
            font-weight: 600 !important;
            font-size: 0.86rem !important;
            transition: all 0.2s ease !important;
        }
        .stButton > button:hover {
            border-color: var(--emerald-primary) !important;
            background: #11251A !important;
            color: #ffffff !important;
            box-shadow: 0 4px 18px rgba(0, 0, 0, 0.35) !important;
        }

        div[data-testid="stDownloadButton"] > button {
            background: #0E1D15 !important;
            color: var(--text-primary) !important;
            border: 1px solid var(--border-mid) !important;
            border-radius: 12px !important;
            font-weight: 700 !important;
            transition: all 0.22s ease !important;
        }
        div[data-testid="stDownloadButton"] > button:hover {
            background: #164A33 !important;
            border-color: var(--emerald-bright) !important;
            box-shadow: 0 6px 22px rgba(43, 166, 111, 0.24) !important;
            color: #ffffff !important;
        }

        /* Top Navigation */
        .tm-topbar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin: 4px 0 24px;
            padding-bottom: 14px;
            border-bottom: 1px solid var(--border-subtle);
        }
        .tm-brand {
            display: flex;
            align-items: center;
            gap: 14px;
        }
        .tm-mark {
            width: 44px;
            height: 44px;
            border-radius: 14px;
            display: grid;
            place-items: center;
            background: linear-gradient(135deg, #35B97A, #164A33);
            color: #F4FAF6;
            font-size: 24px;
            box-shadow: 0 0 24px rgba(43, 166, 111, 0.35);
            border: 1px solid rgba(143, 227, 181, 0.25);
        }
        .tm-brand-name {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 1.35rem;
            font-weight: 700;
            letter-spacing: -0.03em;
            color: #ffffff;
        }
        .tm-brand-sub {
            color: var(--text-secondary);
            font-size: 0.78rem;
            margin-top: 1px;
        }
        .tm-topbar-credit {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            color: var(--text-secondary);
            font-size: 0.78rem;
            letter-spacing: 0.01em;
            border: 1px solid var(--border-subtle);
            background: rgba(143, 227, 181, 0.04);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            padding: 8px 15px;
            border-radius: 999px;
            white-space: nowrap;
        }
        .tm-topbar-credit strong {
            color: var(--gold-warm);
            font-weight: 700;
        }

        /* Hero Section */
        .tm-hero {
            position: relative;
            overflow: hidden;
            border: 1px solid var(--border-mid);
            border-radius: var(--radius-xl);
            padding: 3.6rem 3.6rem 3.2rem;
            margin-bottom: 28px;
            background:
                radial-gradient(circle at 82% 20%, rgba(43, 166, 111, 0.16), transparent 40%),
                radial-gradient(circle at 15% 85%, rgba(242, 200, 121, 0.08), transparent 35%),
                linear-gradient(145deg, #0B1710 0%, #050B08 75%);
            box-shadow: 0 22px 52px rgba(0, 0, 0, 0.45);
        }
        .tm-hero:after {
            content: "";
            position: absolute;
            width: 360px;
            height: 360px;
            right: -100px;
            top: -120px;
            border: 1px solid rgba(143, 227, 181, 0.15);
            border-radius: 50%;
            box-shadow: 0 0 0 50px rgba(143, 227, 181, 0.025), 0 0 0 100px rgba(143, 227, 181, 0.015);
            pointer-events: none;
        }
        .tm-kicker {
            color: var(--mint-soft);
            text-transform: uppercase;
            letter-spacing: 0.16em;
            font-size: 0.74rem;
            font-weight: 800;
            display: inline-flex;
            align-items: center;
            gap: 6px;
        }
        .tm-hero-title {
            font-size: clamp(2.5rem, 5vw, 4.4rem);
            line-height: 1.02;
            letter-spacing: -0.05em;
            margin: 0.95rem 0 1.2rem;
            max-width: 800px;
            color: #ffffff;
            font-weight: 700;
        }
        .tm-hero-gradient {
            background: linear-gradient(135deg, #F4FAF6 20%, var(--mint-soft) 60%, var(--emerald-bright) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .tm-hero-subtitle {
            color: var(--text-secondary);
            font-size: 1.05rem;
            line-height: 1.7;
            max-width: 720px;
            margin: 0 0 1.8rem;
        }
        .tm-hero-actions {
            display: flex;
            align-items: center;
            gap: 14px;
            margin-bottom: 2rem;
        }
        .tm-hero-btn-primary {
            display: inline-flex;
            align-items: center;
            padding: 10px 22px;
            background: linear-gradient(135deg, #35B97A, #238B5B);
            color: #050B08 !important;
            font-weight: 800;
            font-size: 0.92rem;
            border-radius: 12px;
            text-decoration: none !important;
            box-shadow: 0 6px 20px rgba(43, 166, 111, 0.28);
            transition: all 0.2s ease;
        }
        .tm-hero-btn-primary:hover {
            transform: translateY(-2px);
            filter: brightness(1.08);
            box-shadow: 0 8px 26px rgba(43, 166, 111, 0.42);
        }
        .tm-hero-btn-secondary {
            display: inline-flex;
            align-items: center;
            padding: 10px 20px;
            background: rgba(143, 227, 181, 0.05);
            border: 1px solid var(--border-mid);
            color: var(--text-primary) !important;
            font-weight: 600;
            font-size: 0.88rem;
            border-radius: 12px;
            text-decoration: none !important;
            transition: all 0.2s ease;
        }
        .tm-hero-btn-secondary:hover {
            background: rgba(143, 227, 181, 0.12);
            border-color: var(--mint-soft);
        }
        .tm-chip-row {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
        }
        .tm-chip {
            padding: 7px 14px;
            border: 1px solid var(--border-subtle);
            border-radius: 999px;
            background: rgba(143, 227, 181, 0.04);
            color: var(--text-secondary);
            font-size: 0.8rem;
            font-weight: 600;
            backdrop-filter: blur(8px);
        }
        .tm-chip-dot {
            width: 6px;
            height: 6px;
            border-radius: 50%;
            background: var(--emerald-bright);
            display: inline-block;
            margin-right: 6px;
        }

        /* Section Headings */
        .tm-section-head {
            display: flex;
            justify-content: space-between;
            align-items: flex-end;
            margin-bottom: 16px;
        }
        .tm-section-title {
            font-size: 1.3rem;
            font-weight: 700;
            letter-spacing: -0.03em;
            color: #ffffff;
        }
        .tm-section-kicker {
            color: var(--text-muted);
            font-size: 0.8rem;
            margin-top: 3px;
        }

        /* Destination Carousel & Cards */
        .tm-destination-card {
            overflow: hidden;
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius-lg);
            background: #0B1710;
            transition: transform 0.24s cubic-bezier(0.16, 1, 0.3, 1), border-color 0.24s ease, box-shadow 0.24s ease;
            min-height: 180px;
            margin-bottom: 8px;
            position: relative;
        }
        .tm-destination-card:hover {
            transform: translateY(-4px);
            border-color: var(--emerald-primary);
            box-shadow: 0 16px 36px rgba(0, 0, 0, 0.45), 0 0 24px rgba(43, 166, 111, 0.15);
        }
        .tm-destination-card-selected {
            border-color: var(--gold-warm) !important;
            box-shadow:
                0 18px 42px rgba(0, 0, 0, 0.45),
                0 0 0 1px var(--gold-warm),
                0 0 30px rgba(242, 200, 121, 0.22) !important;
        }
        .tm-destination-card-selected::before {
            content: "✦ SELECTED";
            position: absolute;
            top: 10px;
            left: 10px;
            z-index: 5;
            background: var(--gold-warm);
            color: #050B08;
            font-size: 0.62rem;
            font-weight: 900;
            padding: 3px 8px;
            border-radius: 999px;
            letter-spacing: 0.08em;
        }
        .tm-destination-photo {
            position: relative;
            height: 120px;
            overflow: hidden;
            background-color: #071A12;
        }
        .tm-destination-photo img {
            width: 100%;
            height: 100%;
            object-fit: cover;
            display: block;
            transition: transform 0.45s ease, filter 0.45s ease;
        }
        .tm-destination-card:hover .tm-destination-photo img {
            transform: scale(1.08);
            filter: saturate(1.1) brightness(1.03);
        }
        .tm-destination-gradient {
            position: absolute;
            inset: 0;
            background: linear-gradient(180deg, rgba(5, 11, 8, 0) 25%, rgba(5, 11, 8, 0.85) 100%);
            pointer-events: none;
        }
        .tm-destination-icon {
            position: absolute;
            left: 10px;
            bottom: 8px;
            z-index: 2;
            font-size: 1.35rem;
            filter: drop-shadow(0 2px 6px rgba(0,0,0,0.7));
        }
        .tm-destination-group {
            position: absolute;
            right: 10px;
            bottom: 9px;
            z-index: 3;
            padding: 4px 9px;
            border-radius: 999px;
            background: rgba(5, 11, 8, 0.78);
            border: 1px solid var(--border-mid);
            color: var(--mint-soft);
            font-size: 0.68rem;
            font-weight: 800;
            backdrop-filter: blur(8px);
        }
        .tm-destination-body {
            padding: 12px 14px 14px;
        }
        .tm-destination-title {
            color: #ffffff;
            font-weight: 800;
            font-size: 0.96rem;
            letter-spacing: -0.02em;
        }
        .tm-destination-sub {
            color: var(--text-secondary);
            font-size: 0.74rem;
            line-height: 1.45;
            margin-top: 4px;
        }
        .tm-destination-destination {
            color: var(--text-muted);
            font-size: 0.7rem;
            margin-top: 6px;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .tm-carousel-status {
            text-align: center;
            color: var(--text-secondary);
            font-size: 0.8rem;
            font-weight: 650;
            padding: 9px 14px;
            border: 1px solid var(--border-subtle);
            border-radius: 999px;
            background: rgba(143, 227, 181, 0.035);
        }

        /* Form Container & Highlight */
        .tm-planner-box {
            background: linear-gradient(180deg, #0B1710 0%, #07100B 100%);
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius-xl);
            padding: 24px;
            transition: all 0.3s ease;
        }
        @keyframes tmFormGlow {
            0% {
                box-shadow: 0 0 0 2px var(--emerald-primary), 0 0 36px rgba(43, 166, 111, 0.45);
                border-color: var(--emerald-bright);
            }
            50% {
                box-shadow: 0 0 0 3px var(--gold-warm), 0 0 32px rgba(242, 200, 121, 0.35);
                border-color: var(--gold-warm);
            }
            100% {
                box-shadow: 0 16px 44px rgba(0, 0, 0, 0.35);
                border-color: var(--border-subtle);
            }
        }
        .tm-planner-highlighted {
            animation: tmFormGlow 2.4s cubic-bezier(0.16, 1, 0.3, 1) forwards;
        }
        .tm-form-group-title {
            color: var(--mint-soft);
            font-size: 0.74rem;
            font-weight: 800;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            margin-bottom: 8px;
            display: flex;
            align-items: center;
            gap: 6px;
        }
        .tm-form-helper {
            margin-top: 10px;
            color: var(--text-muted);
            font-size: 0.78rem;
            line-height: 1.55;
        }

        /* Dedicated Dedicated Planning Screen (Page 2) */
        .tm-planning-screen {
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 2.5rem 0 3.5rem;
        }
        .tm-planning-card {
            background: linear-gradient(180deg, #0E1D15 0%, #07100B 100%);
            border: 1px solid var(--border-mid);
            border-radius: var(--radius-xl);
            padding: 38px 44px;
            max-width: 580px;
            width: 100%;
            text-align: center;
            box-shadow: 0 26px 70px rgba(0, 0, 0, 0.55), 0 0 40px rgba(43, 166, 111, 0.12);
        }
        .tm-planning-badge {
            display: inline-block;
            padding: 5px 12px;
            border-radius: 999px;
            background: rgba(43, 166, 111, 0.14);
            border: 1px solid rgba(143, 227, 181, 0.28);
            color: var(--mint-soft);
            font-size: 0.7rem;
            font-weight: 800;
            letter-spacing: 0.14em;
            margin-bottom: 12px;
        }
        .tm-planning-pulse-ring {
            width: 72px;
            height: 72px;
            border-radius: 22px;
            margin: 0 auto 18px;
            display: grid;
            place-items: center;
            background: rgba(43, 166, 111, 0.14);
            border: 1px solid rgba(53, 185, 122, 0.45);
            box-shadow: 0 0 32px rgba(43, 166, 111, 0.25);
            animation: tmRingPulse 2s infinite ease-in-out;
        }
        @keyframes tmRingPulse {
            0%, 100% { transform: scale(1); box-shadow: 0 0 20px rgba(43, 166, 111, 0.2); }
            50% { transform: scale(1.06); box-shadow: 0 0 38px rgba(43, 166, 111, 0.42); }
        }
        .tm-planning-title {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 1.55rem;
            font-weight: 700;
            color: #ffffff;
            margin-bottom: 6px;
        }
        .tm-planning-sub {
            color: var(--text-secondary);
            font-size: 0.86rem;
            margin-bottom: 22px;
            line-height: 1.55;
        }

        /* Timeline Agent Flow */
        .tm-agent-timeline {
            display: flex;
            flex-direction: column;
            gap: 10px;
            text-align: left;
            position: relative;
            margin-top: 14px;
        }
        .tm-agent-node {
            display: flex;
            align-items: center;
            gap: 14px;
            padding: 12px 16px;
            border-radius: 14px;
            background: #09150F;
            border: 1px solid var(--border-subtle);
            color: var(--text-muted);
            transition: all 0.25s ease;
        }
        .tm-agent-node.completed {
            border-color: rgba(143, 227, 181, 0.25);
            color: var(--text-primary);
            background: rgba(43, 166, 111, 0.05);
        }
        .tm-agent-node.completed .tm-node-dot {
            color: var(--emerald-bright);
            font-weight: 900;
        }
        .tm-agent-node.active {
            border-color: var(--emerald-bright);
            background: rgba(43, 166, 111, 0.12);
            color: #ffffff;
            box-shadow: 0 0 20px rgba(43, 166, 111, 0.2);
        }
        .tm-agent-node.active .tm-node-dot {
            color: var(--emerald-bright);
            animation: tmBlink 0.9s infinite alternate;
        }
        @keyframes tmBlink {
            from { opacity: 0.4; transform: scale(0.9); }
            to { opacity: 1; transform: scale(1.15); }
        }
        .tm-node-dot {
            font-size: 1.05rem;
            width: 22px;
            text-align: center;
        }
        .tm-node-label {
            font-weight: 700;
            font-size: 0.88rem;
            color: inherit;
        }
        .tm-node-desc {
            font-size: 0.76rem;
            color: var(--text-secondary);
            margin-top: 1px;
        }

        /* Human Review Screen (Page 3) */
        .tm-approval-hero {
            padding: 30px 34px;
            border-radius: var(--radius-xl);
            border: 1px solid var(--border-mid);
            background:
                radial-gradient(circle at 88% 20%, rgba(43, 166, 111, 0.14), transparent 32%),
                radial-gradient(circle at 10% 90%, rgba(242, 200, 121, 0.08), transparent 30%),
                linear-gradient(135deg, #0E1D15 0%, #07100B 100%);
            margin: 18px 0 22px;
            box-shadow: 0 22px 50px rgba(0,0,0,0.38);
        }
        .tm-approval-kicker {
            color: var(--mint-soft);
            font-size: 0.74rem;
            font-weight: 900;
            letter-spacing: 0.15em;
        }
        .tm-approval-title {
            color: #ffffff;
            font-family: 'Space Grotesk', sans-serif;
            font-size: clamp(1.8rem, 3.2vw, 2.8rem);
            line-height: 1.05;
            letter-spacing: -0.04em;
            margin: 8px 0;
        }
        .tm-approval-sub {
            color: var(--text-secondary);
            font-size: 0.94rem;
            line-height: 1.62;
            max-width: 780px;
        }

        /* Final Result Blueprint (Page 4) */
        .tm-result-hero {
            padding: 34px;
            border-radius: var(--radius-xl);
            border: 1px solid var(--border-mid);
            background:
                radial-gradient(circle at 90% 20%, rgba(43, 166, 111, 0.15), transparent 32%),
                radial-gradient(circle at 10% 90%, rgba(139, 213, 232, 0.07), transparent 35%),
                linear-gradient(135deg, #0E1D15 0%, #07100B 100%);
            margin-bottom: 24px;
            box-shadow: 0 22px 52px rgba(0, 0, 0, 0.35);
        }
        .tm-result-kicker {
            color: var(--mint-soft);
            text-transform: uppercase;
            letter-spacing: 0.16em;
            font-size: 0.74rem;
            font-weight: 800;
        }
        .tm-result-title {
            font-family: 'Space Grotesk', sans-serif;
            font-size: clamp(2.2rem, 4.5vw, 3.8rem);
            line-height: 1.02;
            margin: 8px 0 14px;
            letter-spacing: -0.045em;
            color: #ffffff;
        }
        .tm-meta {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-bottom: 16px;
        }
        .tm-meta span {
            padding: 8px 14px;
            border-radius: 999px;
            background: #0B1710;
            border: 1px solid var(--border-subtle);
            color: var(--text-secondary);
            font-size: 0.82rem;
            font-weight: 600;
        }
        .tm-route {
            display: inline-flex;
            align-items: center;
            gap: 12px;
            padding: 8px 18px;
            border-radius: 999px;
            background: rgba(43, 166, 111, 0.07);
            border: 1px solid var(--border-mid);
            color: #e5f4ec;
            font-weight: 700;
            font-size: 0.9rem;
        }
        .tm-route-line {
            color: var(--emerald-bright);
            font-size: 1.15rem;
        }

        /* Section Bars */
        .tm-section-bar {
            display: flex;
            align-items: center;
            gap: 12px;
            margin: 30px 0 16px;
        }
        .tm-section-num {
            width: 32px;
            height: 32px;
            border-radius: 10px;
            display: grid;
            place-items: center;
            background: #0B1710;
            color: var(--mint-soft);
            border: 1px solid var(--border-mid);
            font-size: 0.78rem;
            font-weight: 800;
        }
        .tm-section-label {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 1.12rem;
            font-weight: 700;
            letter-spacing: -0.02em;
            color: #ffffff;
        }
        .tm-section-line {
            flex: 1;
            height: 1px;
            background: var(--border-subtle);
        }

        /* Cards & Typography */
        .tm-card {
            background: linear-gradient(180deg, #0B1710, #07100B);
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius-lg);
            padding: 24px;
            box-shadow: 0 14px 34px rgba(0, 0, 0, 0.25);
        }
        .tm-body {
            color: var(--text-secondary);
            line-height: 1.76;
            font-size: 0.95rem;
        }
        .tm-body p {
            margin: 0 0 0.85rem;
        }
        .tm-body p:last-child {
            margin-bottom: 0;
        }

        /* Transport Card */
        .tm-transport-card {
            display: flex;
            align-items: flex-start;
            gap: 18px;
            padding: 24px;
            border: 1px solid var(--border-subtle);
            background: linear-gradient(135deg, #0E1D15, #07100B);
            border-radius: var(--radius-lg);
            box-shadow: 0 14px 34px rgba(0, 0, 0, 0.25);
        }
        .tm-transport-icon {
            width: 50px;
            height: 50px;
            border-radius: 14px;
            display: grid;
            place-items: center;
            background: rgba(43, 166, 111, 0.12);
            border: 1px solid var(--border-mid);
            color: var(--emerald-bright);
            font-size: 1.4rem;
            flex-shrink: 0;
        }
        .tm-transport-title {
            color: #ffffff;
            font-family: 'Space Grotesk', sans-serif;
            font-size: 1.1rem;
            font-weight: 700;
        }
        .tm-transport-sub {
            color: var(--text-secondary);
            font-size: 0.9rem;
            line-height: 1.7;
            margin-top: 8px;
        }

        /* Hotel Cards */
        .tm-hotel-card {
            min-height: 205px;
            padding: 22px;
            border: 1px solid var(--border-subtle);
            background: linear-gradient(180deg, #0E1D15, #07100B);
            border-radius: var(--radius-lg);
            transition: transform 0.22s ease, border-color 0.22s ease;
            box-shadow: 0 14px 34px rgba(0, 0, 0, 0.25);
            margin-bottom: 14px;
        }
        .tm-hotel-card:hover {
            transform: translateY(-3px);
            border-color: var(--emerald-primary);
        }
        .tm-hotel-badge {
            display: inline-block;
            padding: 5px 11px;
            border-radius: 999px;
            background: rgba(43, 166, 111, 0.12);
            border: 1px solid var(--border-mid);
            color: var(--mint-soft);
            font-size: 0.68rem;
            font-weight: 800;
            letter-spacing: 0.08em;
        }
        .tm-hotel-title {
            color: #ffffff;
            font-family: 'Space Grotesk', sans-serif;
            font-size: 1.08rem;
            font-weight: 700;
            margin-top: 8px;
        }
        .tm-hotel-rating {
            color: var(--gold-warm);
            font-size: 0.84rem;
            font-weight: 700;
            margin-top: 6px;
        }
        .tm-hotel-price {
            color: #ffffff;
            font-size: 0.96rem;
            font-weight: 800;
            margin-top: 8px;
        }
        .tm-hotel-body {
            color: var(--text-secondary);
            font-size: 0.85rem;
            line-height: 1.64;
            margin-top: 8px;
        }

        /* Weather UI */
        .tm-stat-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 12px;
            margin-bottom: 14px;
        }
        .tm-stat {
            padding: 18px;
            border-radius: var(--radius-md);
            background: #0B1710;
            border: 1px solid var(--border-subtle);
            text-align: center;
        }
        .tm-stat-value {
            font-size: 1.4rem;
            font-weight: 800;
            color: #ffffff;
            font-family: 'Space Grotesk', sans-serif;
        }
        .tm-stat-label {
            color: var(--text-muted);
            font-size: 0.74rem;
            margin-top: 5px;
            text-transform: uppercase;
            letter-spacing: 0.06em;
        }
        .tm-weather-card {
            padding: 16px;
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius-md);
            background: #0B1710;
            text-align: center;
        }
        .tm-weather-date {
            color: var(--text-muted);
            font-size: 0.74rem;
        }
        .tm-weather-temp {
            color: #ffffff;
            font-size: 1.2rem;
            font-weight: 800;
            margin-top: 6px;
        }
        .tm-weather-cond {
            color: var(--text-secondary);
            font-size: 0.78rem;
            margin-top: 4px;
        }

        /* Day-by-Day Itinerary (Travel Journal Aesthetic) */
        .tm-day-card-v2 {
            background: linear-gradient(180deg, #0E1D15 0%, #07100B 100%);
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius-xl);
            padding: 26px 30px;
            margin: 0 0 20px;
            box-shadow: 0 16px 42px rgba(0, 0, 0, 0.28);
            transition: border-color 0.22s ease;
        }
        .tm-day-card-v2:hover {
            border-color: var(--border-mid);
        }
        .tm-day-top {
            display: flex;
            justify-content: space-between;
            gap: 12px;
            align-items: center;
        }
        .tm-day-number {
            color: var(--mint-soft);
            font-size: 0.78rem;
            font-weight: 900;
            letter-spacing: 0.14em;
        }
        .tm-day-date {
            color: var(--text-secondary);
            font-size: 0.84rem;
            font-weight: 500;
        }
        .tm-day-title-v2 {
            color: #ffffff;
            font-family: 'Space Grotesk', sans-serif;
            font-size: 1.38rem;
            font-weight: 700;
            letter-spacing: -0.02em;
            margin-top: 6px;
        }
        .tm-day-count {
            color: var(--text-muted);
            font-size: 0.8rem;
            margin: 6px 0 22px;
        }
        .tm-timeline {
            border-left: 1px solid rgba(43, 166, 111, 0.25);
            margin-left: 12px;
            padding-left: 22px;
        }
        .tm-timeline-row {
            display: grid;
            grid-template-columns: 46px 95px 1fr;
            gap: 14px;
            position: relative;
            padding: 13px 0;
            border-bottom: 1px solid rgba(143, 227, 181, 0.05);
        }
        .tm-timeline-row:last-child {
            border-bottom: none;
        }
        .tm-timeline-row:before {
            content: "";
            position: absolute;
            left: -26px;
            top: 24px;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--emerald-bright);
            box-shadow: 0 0 12px rgba(43, 166, 111, 0.6);
        }
        .tm-timeline-num {
            color: var(--text-muted);
            font-size: 0.74rem;
            font-weight: 700;
            padding-top: 2px;
        }
        .tm-timeline-time {
            color: var(--mint-soft);
            font-size: 0.84rem;
            font-weight: 800;
            padding-top: 1px;
        }
        .tm-timeline-activity {
            color: #ffffff;
            font-weight: 700;
            font-size: 0.96rem;
            line-height: 1.48;
        }
        .tm-timeline-detail {
            color: var(--text-secondary);
            font-size: 0.84rem;
            line-height: 1.62;
            margin-top: 4px;
        }

        /* Budget Card */
        .tm-budget-card {
            background: linear-gradient(180deg, #0B1710, #07100B);
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius-lg);
            padding: 26px;
            box-shadow: 0 14px 34px rgba(0, 0, 0, 0.25);
        }
        .tm-budget-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 13px 0;
            border-bottom: 1px solid rgba(143, 227, 181, 0.06);
        }
        .tm-budget-row:last-child {
            border-bottom: none;
        }
        .tm-budget-row span {
            color: var(--text-secondary);
            font-size: 0.92rem;
        }
        .tm-budget-row strong {
            color: #ffffff;
            font-size: 0.98rem;
            font-weight: 700;
        }
        .tm-budget-total-box {
            margin-top: 20px;
            padding: 20px 22px;
            border-radius: var(--radius-md);
            border: 1px solid var(--border-mid);
            background: linear-gradient(135deg, rgba(43, 166, 111, 0.12) 0%, rgba(43, 166, 111, 0.03) 100%);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .tm-budget-total-label {
            color: var(--text-secondary);
            font-size: 0.88rem;
            font-weight: 600;
        }
        .tm-budget-total-value {
            color: var(--gold-warm);
            font-family: 'Space Grotesk', sans-serif;
            font-size: 1.55rem;
            font-weight: 800;
        }
        .tm-budget-progress-track {
            height: 6px;
            border-radius: 999px;
            background: rgba(143, 227, 181, 0.08);
            overflow: hidden;
            margin-top: 14px;
        }
        .tm-budget-progress-bar {
            height: 100%;
            border-radius: 999px;
            background: linear-gradient(90deg, #2BA66F, #F2C879);
        }

        /* Travel Notes */
        .tm-note-card {
            padding: 20px 22px;
            margin-bottom: 12px;
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius-md);
            background: #0B1710;
        }
        .tm-note-title {
            color: #ffffff;
            font-weight: 800;
            font-size: 0.94rem;
            margin-bottom: 6px;
        }
        .tm-note-body {
            color: var(--text-secondary);
            font-size: 0.88rem;
            line-height: 1.68;
        }

        /* Footer & Sidebar */
        .tm-footer {
            color: var(--text-muted);
            font-size: 0.76rem;
            text-align: center;
            margin-top: 3.8rem;
            padding-top: 1.4rem;
            border-top: 1px solid var(--border-subtle);
        }
        .tm-side-brand {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 1.25rem;
            font-weight: 700;
            color: #ffffff;
            margin-bottom: 4px;
        }
        .tm-side-copy {
            color: var(--text-secondary);
            font-size: 0.82rem;
            line-height: 1.55;
            margin-bottom: 18px;
        }
        .tm-side-chip {
            display: inline-block;
            padding: 6px 11px;
            margin: 0 6px 6px 0;
            border: 1px solid var(--border-subtle);
            border-radius: 999px;
            background: rgba(143, 227, 181, 0.04);
            color: var(--text-secondary);
            font-size: 0.74rem;
            font-weight: 500;
        }
        .tm-side-tech {
            color: var(--text-secondary);
            font-size: 0.82rem;
            padding: 7px 0;
        }
        .tm-side-credit {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 10px;
            padding: 12px 14px;
            border: 1px solid var(--border-subtle);
            border-radius: 14px;
            background: rgba(143, 227, 181, 0.04);
            color: var(--text-muted);
            font-size: 0.76rem;
        }
        .tm-side-credit strong {
            color: var(--gold-warm);
        }

        /* Responsive Media Queries */
        @media (max-width: 1024px) {
            .block-container {
                padding: 1rem 1.4rem 3rem !important;
            }
            .tm-hero {
                padding: 2.8rem 2.4rem 2.4rem !important;
            }
            .tm-stat-grid {
                grid-template-columns: repeat(2, 1fr) !important;
            }
        }

        @media (max-width: 768px) {
            .tm-hero-title {
                font-size: 2.6rem !important;
            }
            .tm-hero {
                padding: 2.2rem 1.6rem 2rem !important;
                border-radius: 20px !important;
            }
            .tm-topbar-credit {
                font-size: 0.72rem !important;
                padding: 6px 11px !important;
            }
            .tm-timeline-row {
                grid-template-columns: 36px 82px 1fr !important;
                gap: 8px !important;
            }
        }

        @media (max-width: 520px) {
            .block-container {
                padding: 0.8rem 0.9rem 2.5rem !important;
            }
            .tm-topbar {
                flex-direction: column !important;
                align-items: flex-start !important;
                gap: 10px !important;
            }
            .tm-brand-sub {
                display: none !important;
            }
            .tm-hero {
                padding: 1.8rem 1.2rem 1.6rem !important;
                border-radius: 18px !important;
            }
            .tm-hero-title {
                font-size: 2.1rem !important;
            }
            .tm-stat-grid {
                grid-template-columns: 1fr 1fr !important;
            }
            .tm-transport-card {
                flex-direction: column !important;
            }
            .tm-timeline-row {
                grid-template-columns: 1fr !important;
                gap: 4px !important;
                padding-left: 6px !important;
            }
            .tm-timeline-row:before {
                left: -23px !important;
                top: 14px !important;
            }
            .tm-timeline-num {
                display: none !important;
            }
            .tm-day-card-v2 {
                padding: 18px 16px !important;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────
# State Initialization & Helpers
# ─────────────────────────────────────────────────────────────────────────
def init_session_state() -> None:
    defaults = {
        "thread_id": str(uuid.uuid4()),
        "plan": None,
        "generation_error": None,
        "dest_input": "",
        "origin_input": "Hyderabad, India",
        "start_date_input": date.today() + timedelta(days=30),
        "end_date_input": date.today() + timedelta(days=37),
        "travelers_input": 2,
        "include_budget_input": True,
        "budget_input": 200000,
        "prefs_input": ["Sightseeing", "Food"],
        "additional_input": "",
        "group_type_input": "Friends",
        "show_all_destinations": False,
        "selected_example": None,
        "carousel_start": 0,
        "scroll_target": None,
        "form_highlight": False,
        "pending_approval": None,
        "view": "home",
        "pending_query": None,
        "pending_trip_values": None,
        "processing_title": "Crafting your travel blueprint",
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def apply_example(name: str) -> None:
    """Load complete preset and trigger smooth scroll + form highlight."""
    ex = EXAMPLE_PROMPTS[name]
    st.session_state.update({
        "dest_input": ex["destination"],
        "origin_input": ex["origin"],
        "start_date_input": date.today() + timedelta(days=30),
        "end_date_input": date.today() + timedelta(days=30 + ex["days"] - 1),
        "travelers_input": ex["travelers"],
        "include_budget_input": True,
        "budget_input": ex["budget"],
        "prefs_input": ex["preferences"],
        "additional_input": ex["additional"],
        "group_type_input": EXAMPLE_GROUP_TYPES.get(name, "Friends"),
        "selected_example": name,
        "scroll_target": "tm-trip-form",
        "form_highlight": True,
    })


def reset_trip() -> None:
    st.session_state["plan"] = None
    st.session_state["generation_error"] = None
    st.session_state["show_all_destinations"] = False
    st.session_state["selected_example"] = None
    st.session_state["pending_approval"] = None
    st.session_state["scroll_target"] = None
    st.session_state["form_highlight"] = False
    st.session_state["carousel_start"] = 0
    st.session_state["view"] = "home"
    st.session_state["pending_query"] = None
    st.session_state["pending_trip_values"] = None
    st.session_state["thread_id"] = str(uuid.uuid4())


def format_inr(amount, symbol: str = "₹") -> str:
    try:
        n = int(round(float(amount)))
    except (TypeError, ValueError):
        return str(amount)
    sign = "-" if n < 0 else ""
    n = abs(n)
    text = str(n)
    if len(text) <= 3:
        grouped = text
    else:
        last3, rest = text[-3:], text[:-3]
        parts = []
        while len(rest) > 2:
            parts.insert(0, rest[-2:])
            rest = rest[:-2]
        if rest:
            parts.insert(0, rest)
        grouped = ",".join(parts) + "," + last3
    return f"{sign}{symbol}{grouped}"


def try_parse_structured(raw):
    if raw is None:
        return None
    if not isinstance(raw, str):
        return raw
    text = raw.strip()
    if not text:
        return None
    for parser in (json.loads, ast.literal_eval):
        try:
            return parser(text)
        except Exception:
            continue
    return None


def parse_weather_results(raw: str):
    if not raw or not isinstance(raw, str):
        return None, None
    current_match = re.search(r"Current Weather:\s*(.*?)\s*Forecast:", raw, re.S | re.I)
    forecast_match = re.search(r"Forecast:\s*(.*)$", raw, re.S)
    current_raw = current_match.group(1).strip() if current_match else None
    forecast_raw = forecast_match.group(1).strip() if forecast_match else None
    current = try_parse_structured(current_raw) if current_raw else None
    forecast = try_parse_structured(forecast_raw) if forecast_raw else None
    return current if current is not None else current_raw, forecast if forecast is not None else forecast_raw


_HEADER_RE = re.compile(r"^\s*#{1,4}\s*(.+?)\s*#*\s*$|^\s*\*\*(.+?)\*\*\s*$")
_DAY_RE = re.compile(r"^\s*day\s*\d+\b", re.I)
_BUDGET_RE = re.compile(r"budget|cost|estimate", re.I)
_TIPS_RE = re.compile(r"\btip|advice|note|recommend", re.I)


def split_itinerary_sections(text: str):
    """Convert common LLM itinerary formats into predictable named sections."""
    if not text or not str(text).strip():
        return OrderedDict()

    sections = OrderedDict()
    current = "Overview"
    sections[current] = []

    heading_re = re.compile(
        r"^\s*(?:#{1,6}\s*)?(?:\*\*)?\s*(.+?)\s*(?:\*\*)?\s*$",
        re.I,
    )
    day_re = re.compile(r"^\s*(?:day|d)\s*[-#: ]?\s*(\d{1,2})\b(?:\s*[-:–—]\s*(.*))?", re.I)

    def clean_heading(value: str) -> str:
        value = re.sub(r"^[-*•]+\s*", "", value).strip()
        value = value.strip("#* ")
        return value

    for raw_line in str(text).strip().splitlines():
        line = raw_line.strip()
        if not line:
            sections.setdefault(current, []).append("")
            continue

        dm = day_re.match(line)
        if dm:
            number = dm.group(1)
            label = (dm.group(2) or "").strip()
            current = f"Day {number}" + (f" — {label}" if label else "")
            sections.setdefault(current, [])
            continue

        hm = heading_re.match(line)
        candidate = clean_heading(hm.group(1)) if hm else ""
        is_heading = bool(candidate) and len(candidate) <= 90 and (
            line.startswith("#")
            or line.startswith("**")
            or bool(re.match(r"^(overview|summary|budget|estimated budget|cost|expenses|tips?|travel notes|flight|flights|transport|hotels?|accommodation|weather)\b", candidate, re.I))
        )
        if is_heading:
            current = candidate
            sections.setdefault(current, [])
            continue

        sections.setdefault(current, []).append(raw_line)

    return OrderedDict(
        (title, "\n".join(body).strip())
        for title, body in sections.items()
        if "\n".join(body).strip()
    )


def categorize_sections(sections):
    """Bucket sections while preserving every piece of itinerary text."""
    days, budget, tips, overview = OrderedDict(), OrderedDict(), OrderedDict(), OrderedDict()
    for title, body in sections.items():
        if re.match(r"^day\s*\d+\b", title, re.I):
            days[title] = body
        elif _BUDGET_RE.search(title):
            budget[title] = body
        elif _TIPS_RE.search(title):
            tips[title] = body
        else:
            overview[title] = body
    return overview, days, budget, tips


def build_user_query(destination, origin, start_dt, end_dt, travelers, budget, preferences, additional, group_type="") -> str:
    """Build a planning request with strict presentation guidelines."""
    duration = (end_dt - start_dt).days + 1 if (start_dt and end_dt) else None

    parts = [
        f"Plan a {duration}-day trip to {destination}"
        if duration else f"Plan a trip to {destination}"
    ]
    if origin:
        parts.append(f"departing from {origin}")
    if start_dt and end_dt:
        parts.append(
            f"traveling from {start_dt.strftime('%d %b %Y')} "
            f"to {end_dt.strftime('%d %b %Y')}"
        )
    if travelers:
        parts.append(
            f"for {travelers} traveler{'s' if travelers != 1 else ''}"
        )
    if group_type:
        parts.append(f"traveling as a {group_type.lower()} group")
    if budget:
        parts.append(f"with a total budget of {format_inr(budget)}")
    if preferences:
        parts.append("Trip preferences: " + ", ".join(preferences))
    if additional and additional.strip():
        parts.append("Additional requirements: " + additional.strip())

    format_contract = f"""

IMPORTANT OUTPUT FORMAT
Return the final travel plan using these exact top-level headings. Do not
change the heading names. Do not put JSON, tool traces, agent names, or
internal reasoning in the answer.

TRIP OVERVIEW
Write a short 2-4 sentence overview of the trip and what makes the plan
suitable for this group.

FLIGHT PLAN
Give the most useful flight/transport guidance available from the researched
flight data. Clearly label route, airline, cabin, stops, timing, and price
when those details are actually available. Never invent missing details.

ACCOMMODATION
Give the best hotel/accommodation options from the available search data.
For each option include name, location/area, why it fits, rating if available,
and price if available.

WEATHER
Summarize current/forecast weather information available for the destination
and explain what the traveler should pack or plan around.

DAY 1
Title the day with a concise theme, then list the recommended activities in
chronological order. Include approximate timing only when useful.

DAY 2
Use the same structure. Continue through DAY {duration or 1}.

BUDGET BREAKDOWN
Show a simple category-based estimate such as flights, accommodation, local
transport, food, activities, and buffer. Use only figures supported by the
available results or clearly mark them as estimates.

TRAVEL TIPS
Give practical booking, transport, weather, safety, or money-saving tips.

Keep the language concise and user-facing. The goal is a polished travel
plan that can be displayed directly in a travel-booking style interface and
exported to PDF.
"""

    return ". ".join(parts) + "." + format_contract


def _render_loading_state(active_step_idx: int) -> str:
    """Render the animated multi-agent timeline with glowing pulses & checks."""
    step_html = []
    for i, (node_key, label, desc) in enumerate(PIPELINE_STEPS):
        if i < active_step_idx:
            status_cls = "completed"
            dot_content = "✓"
            badge = "Completed"
        elif i == active_step_idx:
            status_cls = "active"
            dot_content = "◉"
            badge = "Active"
        else:
            status_cls = "pending"
            dot_content = "○"
            badge = "Pending"

        step_html.append(
            f'<div class="tm-agent-node {status_cls}">'
            f'<span class="tm-node-dot">{dot_content}</span>'
            f'<div style="flex:1;">'
            f'<div class="tm-node-label">{label} <span style="font-size:0.68rem; margin-left:6px; opacity:0.8;">[{badge}]</span></div>'
            f'<div class="tm-node-desc">{desc}</div>'
            f'</div>'
            f'</div>'
        )

    progress_pct = int(min(100, max(14, int(((active_step_idx + 1) / len(PIPELINE_STEPS)) * 100))))
    return f"""
    <div class="tm-planning-screen">
      <div class="tm-planning-card">
        <div class="tm-planning-badge">✦ TRAVELMIND AI MULTI-AGENT SYSTEM</div>
        <div class="tm-planning-pulse-ring">
          <span style="font-size: 32px;">🧭</span>
        </div>
        <div class="tm-planning-title">Building your journey...</div>
        <div class="tm-planning-sub">Autonomous agents are synchronizing transport routes, accommodations, climate patterns, and curated itineraries.</div>
        <div class="tm-budget-progress-track" style="margin-bottom: 24px; height: 5px;">
          <div class="tm-budget-progress-bar" style="width: {progress_pct}%;"></div>
        </div>
        <div class="tm-agent-timeline">
          {''.join(step_html)}
        </div>
      </div>
    </div>
    """


def _scroll_to(anchor_id: str) -> None:
    """Smoothly scroll the main Streamlit parent page to a named anchor."""
    if not anchor_id:
        return

    components.html(
        f"""
        <script>
        (() => {{
            const targetId = {json.dumps(anchor_id)};
            const scrollToTarget = () => {{
                const target = window.parent.document.getElementById(targetId);
                if (target) {{
                    target.scrollIntoView({{ behavior: "smooth", block: "start" }});
                    return true;
                }}
                return false;
            }};

            if (!scrollToTarget()) {{
                setTimeout(scrollToTarget, 100);
                setTimeout(scrollToTarget, 300);
            }}
        }})();
        </script>
        """,
        height=0,
    )


# ─────────────────────────────────────────────────────────────────────────
# Graph Execution & HITL Resuming
# ─────────────────────────────────────────────────────────────────────────
def run_travel_planning(query: str, thread_id: str) -> dict:
    """Stream LangGraph nodes while updating the user-friendly timeline."""
    config = {"configurable": {"thread_id": thread_id}}

    initial_state = {
        "messages": [HumanMessage(content=query)],
        "user_query": query,
        "flight_results": "",
        "hotel_results": "",
        "weather_results": "",
        "budget_results": "",
        "itinerary": "",
        "llm_calls": 0,
    }

    result = {
        "flight_results": "",
        "hotel_results": "",
        "weather_results": "",
        "budget_results": "",
        "itinerary": "",
        "approval_request": "",
        "llm_calls": 0,
    }

    loader_placeholder = st.empty()
    loader_placeholder.markdown(
        '<div id="tm-planning-anchor"></div>' + _render_loading_state(0),
        unsafe_allow_html=True,
    )
    _scroll_to("tm-planning-anchor")

    for chunk in app.stream(
        initial_state,
        config=config,
        stream_mode="updates",
    ):
        for node_name, state_update in chunk.items():
            if node_name == "__interrupt__":
                interrupts = state_update or []
                if interrupts:
                    first_interrupt = interrupts[0]
                    if hasattr(first_interrupt, "value"):
                        result["interrupt"] = first_interrupt.value
                    else:
                        result["interrupt"] = first_interrupt
                continue

            if not isinstance(state_update, dict):
                continue

            if node_name == "supervisor":
                loader_placeholder.markdown(_render_loading_state(0), unsafe_allow_html=True)
            elif node_name == "flight_agent":
                result["flight_results"] = state_update.get("flight_results", "")
                loader_placeholder.markdown(_render_loading_state(1), unsafe_allow_html=True)
            elif node_name == "hotel_agent":
                result["hotel_results"] = state_update.get("hotel_results", "")
                loader_placeholder.markdown(_render_loading_state(2), unsafe_allow_html=True)
            elif node_name == "weather_agent":
                result["weather_results"] = state_update.get("weather_results", "")
                loader_placeholder.markdown(_render_loading_state(3), unsafe_allow_html=True)
            elif node_name == "budget_agent":
                result["budget_results"] = state_update.get("budget_results", "")
                loader_placeholder.markdown(_render_loading_state(4), unsafe_allow_html=True)
            elif node_name == "itinerary_agent":
                result["itinerary"] = state_update.get("itinerary", "")
                result["approval_request"] = state_update.get("approval_request", "")
                loader_placeholder.markdown(_render_loading_state(5), unsafe_allow_html=True)
            elif node_name == "human_approval":
                result["approved"] = state_update.get("approved", False)
                result["human_feedback"] = state_update.get("human_feedback", "")
                loader_placeholder.markdown(_render_loading_state(6), unsafe_allow_html=True)
            elif node_name == "final_response":
                result["final_response"] = state_update.get("final_response", "")

            result["llm_calls"] = state_update.get("llm_calls", result["llm_calls"])

    loader_placeholder.empty()
    return result


def resume_travel_planning(thread_id: str, approved: bool, feedback: str = "") -> dict:
    """Resume a paused HITL checkpoint after user feedback or approval."""
    config = {"configurable": {"thread_id": thread_id}}

    result = {
        "approved": approved,
        "human_feedback": feedback,
        "final_response": "",
        "itinerary": "",
        "budget_results": "",
        "flight_results": "",
        "hotel_results": "",
        "weather_results": "",
        "llm_calls": 0,
    }

    loader_placeholder = st.empty()
    loader_placeholder.markdown(_render_loading_state(5), unsafe_allow_html=True)

    for chunk in app.stream(
        Command(resume={"approved": approved, "feedback": feedback}),
        config=config,
        stream_mode="updates",
    ):
        for node_name, state_update in chunk.items():
            if not isinstance(state_update, dict):
                continue

            if node_name == "final_response":
                result["final_response"] = state_update.get("final_response", "")
            elif node_name == "itinerary_agent":
                result["itinerary"] = state_update.get("itinerary", "")
            elif node_name == "budget_agent":
                result["budget_results"] = state_update.get("budget_results", "")

            result["approved"] = state_update.get("approved", result["approved"])
            result["human_feedback"] = state_update.get("human_feedback", result["human_feedback"])
            result["llm_calls"] = state_update.get("llm_calls", result["llm_calls"])

    loader_placeholder.empty()
    return result


# ─────────────────────────────────────────────────────────────────────────
# Navigation & Sidebars
# ─────────────────────────────────────────────────────────────────────────
def render_topbar() -> None:
    st.markdown(
        f"""
        <div class="tm-topbar">
          <div class="tm-brand">
            <div class="tm-mark">🧭</div>
            <div>
              <div class="tm-brand-name">{APP_NAME}</div>
              <div class="tm-brand-sub">✦ {APP_TAGLINE}</div>
            </div>
          </div>
          <div class="tm-topbar-credit">{CREATOR_CREDIT}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown(
            f"<div class='tm-side-brand'>🧭 {APP_NAME}</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div class='tm-side-copy'>"
            "✦ Autonomous travel design powered by multi-agent reasoning. "
            "Routes, stays, microclimates, and itineraries coordinated in harmony."
            "</div>",
            unsafe_allow_html=True,
        )

        if st.button("➕  New Trip", use_container_width=True):
            reset_trip()
            st.rerun()

        if st.session_state.get("plan") is not None:
            if st.button("↻  Plan Another Journey", use_container_width=True):
                reset_trip()
                st.rerun()

        st.divider()

        st.markdown("### 🎯 Trip Preferences")
        prefs = st.session_state.get("prefs_input", [])
        if prefs:
            st.markdown(
                "".join(
                    f"<span class='tm-side-chip'>{p}</span>"
                    for p in prefs
                ),
                unsafe_allow_html=True,
            )
        else:
            st.caption("Select your travel style to personalize.")

        st.divider()

        st.markdown("### 🧩 System Architecture")
        for item in [
            "🧠 LangGraph Multi-Agent Engine",
            "🔌 MCP Distributed Connectors",
            "⚡ Groq High-Speed LLM",
            "🗄️ PostgreSQL State Checkpoints",
        ]:
            st.markdown(
                f"<div class='tm-side-tech'>{item}</div>",
                unsafe_allow_html=True,
            )

        st.divider()

        st.markdown(
            f"<div class='tm-side-credit'>"
            f"<span>✦ Architected by</span><strong>Prem Kumar</strong>"
            f"</div>",
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────────────────
# Page 1: Discover — Hero, Carousel, Planner Form
# ─────────────────────────────────────────────────────────────────────────
def render_hero() -> None:
    st.markdown(
        """
        <section class="tm-hero">
          <div class="tm-kicker">✦ TRAVELMIND AI · MULTI-AGENT INTELLIGENCE</div>
          <h1 class="tm-hero-title">
            Plan Smarter.<br>
            <span class="tm-hero-gradient">Travel Better.</span>
          </h1>
          <p class="tm-hero-subtitle">
            Tell us where you want to go. TravelMind AI turns your idea into a personalized journey, orchestrating flights, curated stays, weather forecasts, and day-by-day itineraries.
          </p>
          <div class="tm-hero-actions">
            <a href="#tm-trip-form" class="tm-hero-btn-primary" onclick="window.parent.document.getElementById('tm-trip-form')?.scrollIntoView({behavior: 'smooth'}); return false;">✦ Start Planning</a>
            <a href="#tm-carousel-anchor" class="tm-hero-btn-secondary" onclick="window.parent.document.getElementById('tm-carousel-anchor')?.scrollIntoView({behavior: 'smooth'}); return false;">Explore destinations ↓</a>
          </div>
          <div class="tm-chip-row">
            <span class="tm-chip"><span class="tm-chip-dot"></span>✈️ Flight Routing</span>
            <span class="tm-chip"><span class="tm-chip-dot"></span>🏨 Curated Stays</span>
            <span class="tm-chip"><span class="tm-chip-dot"></span>🌦️ Microclimate Forecasts</span>
            <span class="tm-chip"><span class="tm-chip-dot"></span>💰 Budget Optimization</span>
            <span class="tm-chip"><span class="tm-chip-dot"></span>🗺️ Day-by-Day Blueprint</span>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_examples() -> None:
    """Render the 20 curated destinations as a left-to-right carousel."""
    total = len(DESTINATION_CARDS)
    page_size = 5
    max_start = max(0, total - page_size)
    current_start = min(max(0, st.session_state.get("carousel_start", 0)), max_start)
    visible_cards = DESTINATION_CARDS[current_start:current_start + page_size]

    st.markdown('<div id="tm-carousel-anchor"></div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="tm-section-head">'
        '<div>'
        '<div class="tm-section-title">🌍 Trending Destinations & Inspiration</div>'
        '<div class="tm-section-kicker">Swipe through 20 curated journeys — tap any card to instantly load its complete itinerary preset</div>'
        '</div>'
        f'<div class="tm-section-kicker">{total} Curated Journeys</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    cols = st.columns(5, gap="small")

    for col, (icon, title, subtitle, destination, image_url, prompt_key) in zip(cols, visible_cards):
        with col:
            group_type = EXAMPLE_GROUP_TYPES.get(prompt_key, "Friends")
            selected = (st.session_state.get("selected_example") == prompt_key)
            selected_class = " tm-destination-card-selected" if selected else ""
            fallback = "https://images.unsplash.com/photo-1501785888041-af3ef285b470?auto=format&fit=crop&w=1200&q=85"

            st.markdown(
                f"""
                <div class="tm-destination-card{selected_class}">
                  <div class="tm-destination-photo">
                    <img src="{image_url}"
                         alt="{title}"
                         loading="lazy"
                         onerror="this.onerror=null;this.src='{fallback}';">
                    <div class="tm-destination-icon">{icon}</div>
                    <div class="tm-destination-gradient"></div>
                    <div class="tm-destination-group">👥 {group_type}</div>
                  </div>
                  <div class="tm-destination-body">
                    <div class="tm-destination-title">{title}</div>
                    <div class="tm-destination-sub">{subtitle}</div>
                    <div class="tm-destination-destination">📍 {destination}</div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.button(
                "✦ Select Journey",
                key=f"dest_{prompt_key}",
                on_click=apply_example,
                args=(prompt_key,),
                use_container_width=True,
            )

    left, mid, right = st.columns([1, 2.4, 1])

    with left:
        if st.button("← Previous", key="dest_prev", use_container_width=True, disabled=current_start <= 0):
            st.session_state["carousel_start"] = max(0, current_start - 3)
            st.rerun()

    with mid:
        st.markdown(
            f"""
            <div class="tm-carousel-status">
                ✦ Showing {current_start + 1}–{min(current_start + page_size, total)} of {total} destinations · Use arrows to scroll
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        if st.button("Next →", key="dest_next", use_container_width=True, disabled=current_start >= max_start):
            st.session_state["carousel_start"] = min(max_start, current_start + 3)
            st.rerun()

    st.markdown(
        '<div class="tm-form-helper" style="text-align:center; margin-top:8px;">'
        '💡 Spiritual & Temple escapes auto-configure as <b>Family</b> · '
        'Mountain/adventure road trips configure as <b>Friends</b> · '
        'Romantic escapes configure as <b>Couple</b>.'
        '</div>',
        unsafe_allow_html=True,
    )


def render_input_form():
    """Render the editable trip planner with highlighting."""
    highlight_class = " tm-planner-highlighted" if st.session_state.get("form_highlight") else ""
    selected_preset = st.session_state.get("selected_example")
    kicker_note = f"Preset active: <b>{selected_preset}</b> · Adjust any detail or click Plan My Trip" if selected_preset else "Start with a preset above or customize every single detail"

    st.markdown('<div id="tm-trip-form"></div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="tm-section-head" style="margin-top:30px;">
          <div>
            <div class="tm-section-title">🧭 Your trip, your way.</div>
            <div class="tm-section-kicker">{kicker_note}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(f'<div class="tm-planner-box{highlight_class}">', unsafe_allow_html=True)

    with st.form("trip_form", clear_on_submit=False):
        st.markdown('<div class="tm-form-group-title">📍 WHERE ARE YOU HEADING?</div>', unsafe_allow_html=True)
        c1, c2 = st.columns([1.35, 1])
        with c1:
            destination = st.text_input("Destination", key="dest_input", placeholder="Kerala, India")
        with c2:
            origin = st.text_input("Departure City", key="origin_input", placeholder="Hyderabad, India")

        st.markdown('<div class="tm-form-group-title" style="margin-top:16px;">👥 TRAVEL GROUP & 📅 DATES</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            group_type = st.selectbox("Travel Group Profile", ["Solo", "Couple", "Family", "Friends"], key="group_type_input")
        with c2:
            start_dt = st.date_input("Start Date", key="start_date_input")
        with c3:
            end_dt = st.date_input("End Date", key="end_date_input")

        c1, c2 = st.columns([1, 1.3])
        with c1:
            travelers = st.number_input("Total Travelers", min_value=1, max_value=25, step=1, key="travelers_input")
        with c2:
            st.caption("👥 Group presets auto-adjust duration, budget targets, accommodation size, and pacing.")

        st.markdown('<div class="tm-form-group-title" style="margin-top:16px;">💰 BUDGET & 🎯 TRAVEL STYLE</div>', unsafe_allow_html=True)
        c1, c2 = st.columns([1, 1.4])
        with c1:
            include_budget = st.checkbox("Set Target Budget Ceiling", key="include_budget_input")
            budget = st.number_input("Budget (₹)", min_value=0, step=5000, key="budget_input", disabled=not include_budget)
        with c2:
            preferences = st.multiselect("Travel Style & Interests", PREFERENCE_OPTIONS, key="prefs_input", placeholder="Sightseeing, food, nature…")

        st.markdown('<div class="tm-form-group-title" style="margin-top:16px;">📝 ADDITIONAL CUSTOMIZATIONS</div>', unsafe_allow_html=True)
        additional = st.text_area(
            "Specific preferences or constraints",
            key="additional_input",
            height=105,
            placeholder="E.g., vegetarian meals, relaxed mornings, temple darshan priorities, Himalayan bike rental, boutique hotels…",
        )

        st.markdown('<div class="tm-primary" style="margin-top:20px;">', unsafe_allow_html=True)
        submitted = st.form_submit_button("✦ Plan My Trip", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    return submitted, dict(
        destination=destination.strip(),
        origin=origin.strip(),
        start_dt=start_dt,
        end_dt=end_dt,
        travelers=int(travelers),
        group_type=group_type,
        budget=(budget if include_budget and budget else None),
        preferences=preferences,
        additional=additional,
    )


def render_input_screen() -> None:
    render_hero()
    render_examples()
    submitted, values = render_input_form()

    # Smooth scroll down to form after preset selection
    if st.session_state.get("scroll_target") == "tm-trip-form":
        _scroll_to("tm-trip-form")
        st.session_state["scroll_target"] = None

    if not submitted:
        return
    if not values["destination"]:
        st.warning("Please enter a destination to begin.")
        return
    if values["end_dt"] < values["start_dt"]:
        st.warning("End date must be on or after the start date.")
        return

    query = build_user_query(
        values["destination"],
        values["origin"],
        values["start_dt"],
        values["end_dt"],
        values["travelers"],
        values["budget"],
        values["preferences"],
        values["additional"],
        values["group_type"],
    )

    # Transition to dedicated Page 2 Planning View
    st.session_state["pending_query"] = query
    st.session_state["pending_trip_values"] = values
    st.session_state["generation_error"] = None
    st.session_state["form_highlight"] = False
    st.session_state["view"] = "processing"
    st.rerun()


# ─────────────────────────────────────────────────────────────────────────
# Page 2: Dedicated Planning Experience
# ─────────────────────────────────────────────────────────────────────────
def render_processing_screen() -> None:
    """Run the multi-agent graph with live timeline updates."""
    query = st.session_state.get("pending_query")
    values = st.session_state.get("pending_trip_values")

    if not query or not values:
        st.session_state["view"] = "home"
        st.rerun()
        return

    try:
        result = run_travel_planning(query, st.session_state["thread_id"])

        st.session_state["plan"] = {
            "destination": values["destination"],
            "origin": values["origin"],
            "start_date": values["start_dt"],
            "end_date": values["end_dt"],
            "duration_days": (values["end_dt"] - values["start_dt"]).days + 1,
            "travelers": values["travelers"],
            "budget": values["budget"],
            "preferences": values["preferences"],
            "additional": values["additional"],
            "group_type": values["group_type"],
            "query": query,
            "generated_at": datetime.now(),
            **result,
        }

        st.session_state["pending_query"] = None
        st.session_state["pending_trip_values"] = None
        st.session_state["generation_error"] = None

        if result.get("interrupt"):
            st.session_state["pending_approval"] = result["interrupt"]
            st.session_state["view"] = "approval"
        else:
            st.session_state["pending_approval"] = None
            st.session_state["view"] = "result"

        st.rerun()

    except Exception:
        logger.exception("Trip planning failed")
        st.session_state["generation_error"] = "planning_failed"
        st.session_state["plan"] = None
        st.session_state["pending_query"] = None
        st.session_state["pending_trip_values"] = None
        st.session_state["view"] = "home"
        st.error("TravelMind could not complete this journey. Please check your travel services and try again.")


# ─────────────────────────────────────────────────────────────────────────
# Page 3: Human-in-the-Loop Review Screen
# ─────────────────────────────────────────────────────────────────────────
def markdown_lite_clean(text: str) -> str:
    if text is None:
        return ""
    return str(text).strip()


def _to_html_paragraphs(text: str) -> str:
    if not text:
        return ""
    text = text.replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    return "".join(f"<p>{p.replace(chr(10), '<br>')}</p>" for p in re.split(r"\n\s*\n", text) if p.strip())


def render_approval_screen(plan: dict) -> None:
    """Show the drafted itinerary for user verification before finalizing."""
    st.markdown('<div id="tm-approval-anchor"></div>', unsafe_allow_html=True)
    _scroll_to("tm-approval-anchor")

    destination = plan.get("destination", "Your Journey")
    st.markdown(
        f"""
        <div class="tm-approval-hero">
          <div class="tm-approval-kicker">🧑💻 HUMAN-IN-THE-LOOP VERIFICATION</div>
          <div class="tm-approval-title">Your journey is ready. ✦</div>
          <div class="tm-approval-sub">
            TravelMind AI has crafted a personalized draft itinerary for <b>{destination}</b>.
            Review the plan below, approve it to finalize, or request specific modifications.
          </div>
          <div class="tm-meta" style="margin-top: 16px; margin-bottom: 0;">
            <span>📍 {destination}</span>
            <span>⏱️ {plan.get('duration_days', '—')} Days</span>
            <span>👥 {plan.get('travelers', '—')} Travelers ({plan.get('group_type', 'Travelers')})</span>
            {f"<span>💰 {format_inr(plan['budget'])}</span>" if plan.get('budget') else ''}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    draft = plan.get("itinerary", "")
    approval_request = plan.get("approval_request", "")
    review_text = approval_request or draft

    st.markdown(
        f"<div class='tm-card tm-body'>{_to_html_paragraphs(review_text)}</div>",
        unsafe_allow_html=True,
    )

    st.markdown('<div style="margin-top:20px; font-weight:700; color:var(--text-secondary); font-size:0.88rem;">💬 Modification Notes (Optional)</div>', unsafe_allow_html=True)
    feedback = st.text_area(
        "Feedback",
        key="hitl_feedback_input",
        height=100,
        placeholder="E.g., Reduce travel time, add more temple visits, use cheaper hotels, or make Day 1 more relaxed…",
        label_visibility="collapsed",
    )

    left, right = st.columns(2)
    with left:
        if st.button("✅ Approve Itinerary", use_container_width=True, type="primary"):
            result = resume_travel_planning(
                st.session_state["thread_id"],
                approved=True,
                feedback=feedback.strip(),
            )
            st.session_state["plan"].update(result)
            st.session_state["pending_approval"] = None
            st.session_state["view"] = "result"
            st.rerun()

    with right:
        if st.button("↻ Request Changes", use_container_width=True):
            result = resume_travel_planning(
                st.session_state["thread_id"],
                approved=False,
                feedback=feedback.strip(),
            )
            st.session_state["plan"].update(result)
            if result.get("interrupt"):
                st.session_state["pending_approval"] = result["interrupt"]
                st.session_state["view"] = "approval"
            else:
                st.session_state["pending_approval"] = None
                st.session_state["view"] = "result"
            st.rerun()


# ─────────────────────────────────────────────────────────────────────────
# Page 4: Final Blueprint Experience & Renderers
# ─────────────────────────────────────────────────────────────────────────
def section_bar(num, label):
    st.markdown(
        f"<div class='tm-section-bar'>"
        f"<div class='tm-section-num'>{num}</div>"
        f"<div class='tm-section-label'>{label}</div>"
        f"<div class='tm-section-line'></div>"
        f"</div>",
        unsafe_allow_html=True,
    )


def _extract_money(text: str):
    if not text:
        return None
    matches = re.findall(
        r"(?:₹|INR\s*)\s*([0-9][0-9,]*(?:\.[0-9]+)?)",
        str(text),
        flags=re.I,
    )
    if not matches:
        return None
    try:
        return float(matches[0].replace(",", ""))
    except ValueError:
        return None


def _strip_md(text: str) -> str:
    if text is None:
        return ""
    value = str(text).replace("<br>", " ").replace("<br/>", " ")
    value = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", value)
    value = re.sub(r"[*_`#]+", "", value)
    value = value.replace("•", " ").replace("–", "-").replace("—", "-")
    return re.sub(r"\s+", " ", value).strip()


def _extract_day_activities(body: str):
    activities = []
    for raw in str(body or "").splitlines():
        line = raw.strip()
        if not line or re.fullmatch(r"[-|:\s]+", line):
            continue

        if line.startswith("|") and line.endswith("|"):
            cells = [_strip_md(c) for c in line.strip("|").split("|")]
            if not cells:
                continue
            low = " ".join(c.lower() for c in cells)
            if "time" in low and "activity" in low:
                continue
            cells = [c for c in cells if c]
            if not cells:
                continue
            if len(cells) >= 3:
                activities.append((cells[0], cells[1], " | ".join(cells[2:])))
            elif len(cells) == 2:
                activities.append((cells[0], cells[1], ""))
            else:
                activities.append(("", cells[0], ""))
            continue

        clean = _strip_md(line.lstrip("-*• "))
        if not clean:
            continue

        tm = re.match(
            r"^((?:\d{1,2})(?::\d{2})?\s*(?:AM|PM|am|pm)?(?:\s*-\s*\d{1,2}(?::\d{2})?\s*(?:AM|PM|am|pm)?)?)"
            r"\s*(?::|[-|])\s*(.+)$",
            clean,
        )
        if tm:
            time_value = tm.group(1).strip()
            remainder = tm.group(2).strip()
            parts = [p.strip() for p in re.split(r"\s+\|\s+|\s+–\s+|\s+-\s+", remainder, maxsplit=1)]
            activity = parts[0]
            detail = parts[1] if len(parts) > 1 else ""
            activities.append((time_value, activity, detail))
            continue

        activities.append(("", clean, ""))

    return activities


def _safe_day_title(title: str, index: int) -> str:
    cleaned = _strip_md(title)
    cleaned = re.sub(r"^day\s*\d+\s*[:\-–]?\s*", "", cleaned, flags=re.I).strip()
    return cleaned or f"Day {index}"


def _day_date_label(plan: dict, index: int) -> str:
    start = plan.get("start_date")
    if start:
        try:
            current = start + timedelta(days=index - 1)
            return current.strftime("%a, %d %b %Y")
        except Exception:
            pass
    return f"Day {index}"


ACTIVITY_EMOJIS = [
    (re.compile(r"\b(flight|fly|airport|terminal|landing|departure|airline|plane)\b", re.I), "✈️"),
    (re.compile(r"\b(train|railway|metro|station|transit|shinkansen|subway)\b", re.I), "🚆"),
    (re.compile(r"\b(hotel|check-in|check-out|resort|stay|hostel|villa|inn)\b", re.I), "🏨"),
    (re.compile(r"\b(breakfast|lunch|dinner|dining|food|restaurant|cuisine|eat|snack|tasting|caf[eé]|coffee|tea|bar|street food)\b", re.I), "🍛"),
    (re.compile(r"\b(temple|shrine|mosque|cathedral|church|monastery|heritage|palace|castle|monument|fort|historical|museum|art gallery|ruins)\b", re.I), "🛕"),
    (re.compile(r"\b(beach|coast|ocean|sea|surf|cove|island|bay)\b", re.I), "🏖️"),
    (re.compile(r"\b(hike|trek|mountain|peak|summit|trail|waterfall|nature|safari|wildlife|jungle|national park)\b", re.I), "🏔️"),
    (re.compile(r"\b(shop|market|bazaar|souvenir|mall|boutique)\b", re.I), "🛍️"),
    (re.compile(r"\b(relax|spa|sunset|massage|leisure|stroll|park|walk|garden)\b", re.I), "🌅"),
    (re.compile(r"\b(boat|cruise|ferry|kayak|river|lake)\b", re.I), "⛵"),
    (re.compile(r"\b(tour|sightseeing|viewpoint|explore|city walk|attraction|discover)\b", re.I), "🗺️"),
]


def get_activity_emoji(activity_text: str, detail_text: str = "") -> str:
    combined = f"{activity_text} {detail_text}"
    for pattern, emoji in ACTIVITY_EMOJIS:
        if pattern.search(combined):
            return emoji
    return "✦"


def get_hotel_badge(title: str, snippet: str, rating=None, price=None) -> str:
    text = f"{title} {snippet}".lower()
    if "luxury" in text or "5-star" in text or "resort & spa" in text:
        return "LUXURY PICK"
    if "budget" in text or "affordable" in text or "hostel" in text or (price and price < 3500):
        return "BEST VALUE"
    if "family" in text or "kids" in text or "spacious" in text or "suites" in text:
        return "FAMILY FRIENDLY"
    if "central" in text or "heart of" in text or "near" in text or "station" in text or "downtown" in text:
        return "GREAT LOCATION"
    if rating:
        try:
            val = float(str(rating).split("/")[0])
            if val >= 4.5:
                return "TOP RATED"
        except Exception:
            pass
    return "RECOMMENDED"


def _render_day_card(title: str, body: str, index: int, plan: dict) -> None:
    day_title = _safe_day_title(title, index)
    date_label = _day_date_label(plan, index)
    activities = _extract_day_activities(body)

    if not activities:
        activities = [("", "No activities were returned for this day.", "")]

    rows = []
    for number, (time_value, activity, detail) in enumerate(activities[:12], start=1):
        icon = get_activity_emoji(activity, detail)
        rows.append(
            f"""
            <div class="tm-timeline-row">
                <div class="tm-timeline-num">{number:02d}</div>
                <div class="tm-timeline-time">{time_value or "Flexible"}</div>
                <div class="tm-timeline-copy">
                    <div class="tm-timeline-activity"><span style="margin-right:8px; font-size:1.05rem;">{icon}</span>{activity}</div>
                    {"<div class='tm-timeline-detail'>" + detail + "</div>" if detail else ""}
                </div>
            </div>
            """
        )

    st.markdown(
        f"""
        <div class="tm-day-card-v2">
            <div class="tm-day-top">
                <div class="tm-day-number">DAY {index:02d}</div>
                <div class="tm-day-date">📅 {date_label}</div>
            </div>
            <div class="tm-day-title-v2">{day_title}</div>
            <div class="tm-day-count">🧭 {len(activities[:12])} Curated Highlights</div>
            <div class="tm-timeline">
                {''.join(rows)}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_result_screen(plan: dict) -> None:
    destination = plan.get("destination", "Your Trip")
    start = plan.get("start_date")
    end = plan.get("end_date")
    date_range = (
        f"{start.strftime('%b %d')} – {end.strftime('%b %d, %Y')}"
        if start and end
        else "Dates flexible"
    )

    st.markdown(
        f"""
        <div class='tm-result-hero'>
            <div class='tm-result-kicker'>✦ FINAL TRAVEL BLUEPRINT</div>
            <div class='tm-result-title'>{destination}</div>
            <div class='tm-meta'>
                <span>📅 {date_range}</span>
                <span>⏱️ {plan.get('duration_days', '—')} Days</span>
                <span>👥 {plan.get('travelers', '—')} Travelers</span>
                <span>🫶 {plan.get('group_type', 'Travelers')}</span>
                {f"<span>💰 {format_inr(plan['budget'])}</span>" if plan.get('budget') else ''}
            </div>
            <div class='tm-route'>
                <span>{plan.get('origin') or 'Departure'}</span>
                <span class='tm-route-line'>✈</span>
                <span>{destination}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Action Bar
    a, b, c = st.columns([1, 1, 1], gap="small")
    with a:
        st.download_button(
            "↓ Download Trip PDF",
            data=create_travel_plan_pdf(plan),
            file_name=f"travelmind_{re.sub(r'[^a-z0-9]+', '_', destination.lower()).strip('_')}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
    with b:
        if st.button("↺ Plan Another Trip", use_container_width=True):
            reset_trip()
            st.rerun()
    with c:
        st.caption("✦ Verified Multi-Agent Travel Plan")

    # 01. Trip Overview
    section_bar("01", "Trip Overview")
    overview_sections = split_itinerary_sections(plan.get("itinerary", ""))
    overview, days, budget_sections, tips = categorize_sections(overview_sections)
    overview_text = overview.get("TRIP OVERVIEW") or overview.get("Trip Overview")
    if overview_text:
        st.markdown(
            f"<div class='tm-card tm-body'>{_to_html_paragraphs(overview_text)}</div>",
            unsafe_allow_html=True,
        )
    else:
        prefs = " · ".join(plan.get("preferences", [])) or "Flexible travel style"
        st.markdown(
            f"<div class='tm-card tm-body'><b>Travel Style:</b> {prefs}<br>"
            f"{_to_html_paragraphs(plan.get('additional', ''))}</div>",
            unsafe_allow_html=True,
        )

    # 02. Flights & Transport
    section_bar("02", "Flights & Transport")
    flight_text = markdown_lite_clean(plan.get("flight_results", ""))
    if flight_text:
        st.markdown(
            f"<div class='tm-transport-card'><div class='tm-transport-icon'>✈</div>"
            f"<div><div class='tm-transport-title'>{plan.get('origin') or 'Origin'} → {destination}</div>"
            f"<div class='tm-transport-sub'>{_to_html_paragraphs(flight_text)}</div></div></div>",
            unsafe_allow_html=True,
        )
    else:
        st.info("No flight or transit guidance was returned for this destination.")

    # 03. Stays & Accommodation
    section_bar("03", "Stays & Accommodation")
    raw_hotels = plan.get("hotel_results", "")
    parsed = try_parse_structured(raw_hotels)
    entries = (
        parsed.get("results")
        if isinstance(parsed, dict) and isinstance(parsed.get("results"), list)
        else parsed if isinstance(parsed, list) else None
    )
    if entries:
        cols = st.columns(min(2, max(1, len(entries[:4]))), gap="small")
        for idx, item in enumerate(entries[:4]):
            if not isinstance(item, dict):
                continue
            title = item.get("title") or item.get("name") or "Accommodation option"
            snippet = item.get("content") or item.get("snippet") or item.get("description") or ""
            rating = item.get("rating") or item.get("score")
            price = _extract_money(str(item.get("content", "")) + " " + str(item.get("description", "")))
            badge = get_hotel_badge(title, snippet, rating, price)
            with cols[idx % len(cols)]:
                st.markdown(
                    f"""<div class='tm-hotel-card'>
                    <div class='tm-hotel-badge'>{badge}</div>
                    <div class='tm-hotel-title'>{title}</div>
                    {f"<div class='tm-hotel-rating'>★ {rating}</div>" if rating else ''}
                    {f"<div class='tm-hotel-price'>From {format_inr(price)}</div>" if price else ''}
                    <div class='tm-hotel-body'>{_to_html_paragraphs(str(snippet)[:600])}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )
    else:
        st.markdown(
            f"<div class='tm-card tm-body'>{_to_html_paragraphs(markdown_lite_clean(raw_hotels)) or 'No accommodation recommendations were returned.'}</div>",
            unsafe_allow_html=True,
        )

    # 04. Weather & Packing
    section_bar("04", "Weather & Packing")
    current, forecast = parse_weather_results(plan.get("weather_results", ""))
    if isinstance(current, dict) and "temperature_c" in current:
        stats = [
            (f"{current.get('temperature_c', '—')}°C", "Temperature"),
            (f"{current.get('feels_like_c', '—')}°C", "Feels Like"),
            (f"{current.get('humidity', '—')}%", "Humidity"),
            (f"{current.get('wind_speed', '—')} m/s", "Wind Speed"),
        ]
        st.markdown(
            "<div class='tm-stat-grid'>"
            + "".join(
                f"<div class='tm-stat'><div class='tm-stat-value'>{v}</div>"
                f"<div class='tm-stat-label'>{l}</div></div>"
                for v, l in stats
            )
            + "</div>",
            unsafe_allow_html=True,
        )
        if current.get("condition"):
            st.caption(f"Current conditions: {str(current['condition']).capitalize()}")
    elif current:
        st.markdown(
            f"<div class='tm-card tm-body'>{_to_html_paragraphs(str(current))}</div>",
            unsafe_allow_html=True,
        )

    st.markdown(
        "<div class='tm-note-card' style='margin-bottom:14px;'>"
        "<div class='tm-note-title'>🎒 Recommended Packing & Travel Cues</div>"
        "<div class='tm-note-body'>Comfortable walking footwear, breathable layered apparel, sun & rain protection, portable battery bank, and universal power adapter.</div>"
        "</div>",
        unsafe_allow_html=True,
    )

    if isinstance(forecast, list) and forecast:
        fcols = st.columns(min(5, len(forecast[:5])), gap="small")
        for idx, item in enumerate(forecast[:5]):
            if not isinstance(item, dict):
                continue
            with fcols[idx % len(fcols)]:
                st.markdown(
                    f"<div class='tm-weather-card'><div class='tm-weather-date'>{item.get('datetime','')}</div>"
                    f"<div class='tm-weather-temp'>{item.get('temperature','—')}°C</div>"
                    f"<div class='tm-weather-cond'>{str(item.get('weather','')).capitalize()}</div></div>",
                    unsafe_allow_html=True,
                )

    # 05. Day-by-Day Itinerary
    section_bar("05", "Day-by-Day Itinerary")
    if days:
        for idx, (title, body) in enumerate(days.items(), start=1):
            _render_day_card(title, body, idx, plan)
    else:
        raw_lines = [x for x in str(plan.get("itinerary", "")).splitlines() if x.strip()]
        for idx, line in enumerate(raw_lines[: plan.get("duration_days", 1)], start=1):
            _render_day_card(f"Day {idx}", line, idx, plan)

    # 06. Budget Breakdown
    section_bar("06", "Budget Breakdown")
    st.caption("Planning estimate only — live prices should be verified before booking.")
    if budget_sections:
        budget_rows = []
        total_estimated = 0
        for title, body in budget_sections.items():
            amount = _extract_money(body)
            if amount:
                total_estimated += amount
            budget_rows.append((title, format_inr(amount) if amount else "Estimate"))

        html_rows = "".join(
            f"<div class='tm-budget-row'><span>{_to_html_paragraphs(label)}</span><strong>{amount}</strong></div>"
            for label, amount in budget_rows
        )

        total_budget_html = ""
        if plan.get("budget"):
            user_budget = float(plan["budget"])
            pct = min(100, int((total_estimated / user_budget) * 100)) if user_budget > 0 else 100
            total_budget_html = f"""
            <div class='tm-budget-total-box'>
                <div>
                    <div class='tm-budget-total-label'>Target Trip Budget</div>
                    <div class='tm-budget-total-value'>{format_inr(user_budget)}</div>
                </div>
                <div style='text-align:right;'>
                    <div class='tm-budget-total-label'>Estimated Breakdown</div>
                    <div style='color:#ffffff; font-weight:700;'>{format_inr(total_estimated) if total_estimated else 'Calculated'}</div>
                </div>
            </div>
            <div class='tm-budget-progress-track'>
                <div class='tm-budget-progress-bar' style='width:{pct}%;'></div>
            </div>
            """
        st.markdown(f"<div class='tm-budget-card'>{html_rows}{total_budget_html}</div>", unsafe_allow_html=True)
    elif plan.get("budget"):
        st.markdown(
            f"<div class='tm-budget-card'><div class='tm-budget-total-box'><div>"
            f"<div class='tm-budget-total-label'>Planned Budget Ceiling</div>"
            f"<div class='tm-budget-total-value'>{format_inr(plan['budget'])}</div>"
            f"</div></div></div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<div class='tm-card tm-body'>No budget ceiling was specified for this trip.</div>",
            unsafe_allow_html=True,
        )

    # 07. Travel Notes & Tips
    section_bar("07", "Travel Notes & Practical Tips")
    if tips:
        for title, body in tips.items():
            st.markdown(
                f"<div class='tm-note-card'><div class='tm-note-title'>{title}</div>"
                f"<div class='tm-note-body'>{_to_html_paragraphs(body)}</div></div>",
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            "<div class='tm-card tm-body'>Check passport/visa validity, verify live flight fares, and reconfirm hotel cancellation policies before making reservations.</div>",
            unsafe_allow_html=True,
        )

    st.markdown(
        f"<div class='tm-footer'>{APP_NAME} · {CREATOR_CREDIT} · "
        "Verify live fares and bookings with official carriers.</div>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────
# PDF Export (Emerald & Forest Green Palette)
# ─────────────────────────────────────────────────────────────────────────
def create_travel_plan_pdf(plan: dict) -> bytes:
    """Generate a clean, high-contrast, emerald-accented travel document."""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.lib import colors
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        PageBreak, KeepTogether, HRFlowable
    )
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    buffer = io.BytesIO()

    # Forest Green Palette for PDF
    accent = colors.HexColor("#2BA66F")
    ink = colors.HexColor("#07100B")
    muted = colors.HexColor("#5D7065")
    border = colors.HexColor("#D2DFD8")
    panel = colors.HexColor("#F2F7F4")
    pale = colors.HexColor("#E5F4EC")
    white = colors.white

    font_candidates = [
        r"C:\Windows\Fonts\seguisym.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for font_path in font_candidates:
        try:
            if os.path.exists(font_path):
                pdfmetrics.registerFont(TTFont("TravelSymbols", font_path))
                break
        except Exception:
            continue

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=1.45 * cm,
        rightMargin=1.45 * cm,
        topMargin=1.35 * cm,
        bottomMargin=1.35 * cm,
        title=f"{APP_NAME} — {plan.get('destination', 'Travel Plan')}",
        author=APP_NAME,
    )

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="TMBrand",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=24,
        leading=27,
        textColor=ink,
        spaceAfter=2,
    ))
    styles.add(ParagraphStyle(
        name="TMTag",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=7.5,
        leading=9,
        textColor=colors.HexColor("#2BA66F"),
        tracking=1.3,
        spaceAfter=9,
    ))
    styles.add(ParagraphStyle(
        name="TMHero",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=25,
        leading=28,
        textColor=ink,
        spaceBefore=4,
        spaceAfter=5,
    ))
    styles.add(ParagraphStyle(
        name="TMIntro",
        parent=styles["Normal"],
        fontSize=9.4,
        leading=13.8,
        textColor=muted,
        spaceAfter=10,
    ))
    styles.add(ParagraphStyle(
        name="TMSection",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=13.5,
        leading=16,
        textColor=ink,
        spaceBefore=13,
        spaceAfter=7,
    ))
    styles.add(ParagraphStyle(
        name="TMSub",
        parent=styles["Heading3"],
        fontName="Helvetica-Bold",
        fontSize=10.2,
        leading=13,
        textColor=ink,
        spaceBefore=4,
        spaceAfter=4,
    ))
    styles.add(ParagraphStyle(
        name="TMBody",
        parent=styles["Normal"],
        fontSize=9.0,
        leading=13.3,
        textColor=ink,
        spaceAfter=4,
    ))
    styles.add(ParagraphStyle(
        name="TMSmall",
        parent=styles["Normal"],
        fontSize=7.8,
        leading=10.5,
        textColor=muted,
        spaceAfter=3,
    ))
    styles.add(ParagraphStyle(
        name="TMLabel",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=7.2,
        leading=9,
        textColor=muted,
    ))
    styles.add(ParagraphStyle(
        name="TMValue",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9.0,
        leading=11.5,
        textColor=ink,
    ))
    styles.add(ParagraphStyle(
        name="TMDistro",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9.5,
        leading=12,
        textColor=ink,
    ))

    def P(value, style="TMBody"):
        value = "" if value is None else str(value)
        value = value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        value = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", value)
        value = value.replace("\n", "<br/>")
        value = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", value)
        return Paragraph(value, styles[style])

    def clean_pdf_text(value):
        return _strip_md(value)

    def add_page_number(canvas, doc_obj):
        canvas.saveState()
        canvas.setStrokeColor(border)
        canvas.line(1.45 * cm, 0.95 * cm, A4[0] - 1.45 * cm, 0.95 * cm)
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(muted)
        canvas.drawString(1.45 * cm, 0.62 * cm, f"{APP_NAME} · {CREATOR_CREDIT}")
        canvas.drawRightString(A4[0] - 1.45 * cm, 0.62 * cm, f"Page {doc_obj.page}")
        canvas.restoreState()

    destination = clean_pdf_text(plan.get("destination", "Travel Plan"))
    origin = clean_pdf_text(plan.get("origin", "Flexible departure"))
    start_date = plan.get("start_date")
    end_date = plan.get("end_date")
    date_range = (
        f"{start_date.strftime('%d %b %Y')} — {end_date.strftime('%d %b %Y')}"
        if start_date and end_date else "Dates not specified"
    )
    days_count = plan.get("duration_days", "—")
    travelers = plan.get("travelers", "—")
    budget = plan.get("budget")
    budget_text = format_inr(budget, symbol="INR ") if budget else "Not specified"

    sections = split_itinerary_sections(plan.get("itinerary", ""))
    overview, days, budget_sections, tips = categorize_sections(sections)
    trip_overview = overview.get("TRIP OVERVIEW") or overview.get("Trip Overview") or ""
    if not trip_overview:
        trip_overview = plan.get("additional") or "A personalized travel itinerary designed with autonomous multi-agent intelligence."

    story = [
        Paragraph(APP_NAME, styles["TMBrand"]),
        Paragraph("AUTONOMOUS MULTI-AGENT TRAVEL BLUEPRINT", styles["TMTag"]),
        HRFlowable(width="100%", thickness=1.0, color=border, spaceBefore=1, spaceAfter=10),
        Paragraph(destination, styles["TMHero"]),
        P(clean_pdf_text(trip_overview), "TMIntro"),
    ]

    summary = [
        [P("FROM", "TMLabel"), P("DATES", "TMLabel"), P("DURATION", "TMLabel"), P("TRAVELERS", "TMLabel")],
        [P(origin, "TMValue"), P(date_range, "TMValue"), P(f"{days_count} days", "TMValue"), P(str(travelers), "TMValue")],
    ]
    summary_table = Table(summary, colWidths=[4.0*cm, 5.7*cm, 3.0*cm, 2.3*cm])
    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), panel),
        ("BOX", (0,0), (-1,-1), 0.7, border),
        ("INNERGRID", (0,0), (-1,-1), 0.35, border),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("TOPPADDING", (0,0), (-1,-1), 8),
        ("BOTTOMPADDING", (0,0), (-1,-1), 8),
        ("LEFTPADDING", (0,0), (-1,-1), 8),
        ("RIGHTPADDING", (0,0), (-1,-1), 8),
    ]))
    story.extend([summary_table, Spacer(1, 8)])

    if budget:
        budget_banner = Table([
            [P("TARGET TRIP BUDGET CEILING", "TMLabel"), P(budget_text, "TMDistro")]
        ], colWidths=[11.0*cm, 4.0*cm])
        budget_banner.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,-1), pale),
            ("BOX", (0,0), (-1,-1), 0.8, colors.HexColor("#A8D9C0")),
            ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
            ("ALIGN", (1,0), (1,0), "RIGHT"),
            ("TOPPADDING", (0,0), (-1,-1), 9),
            ("BOTTOMPADDING", (0,0), (-1,-1), 9),
            ("LEFTPADDING", (0,0), (-1,-1), 9),
            ("RIGHTPADDING", (0,0), (-1,-1), 9),
        ]))
        story.extend([budget_banner, Spacer(1, 5)])

    # 01. At a Glance
    story.append(Paragraph("01  •  TRIP AT A GLANCE", styles["TMSection"]))
    glance_rows = [
        [P("Travel Style", "TMLabel"), P(", ".join(plan.get("preferences", [])) or "Flexible")],
        [P("Special Notes", "TMLabel"), P(clean_pdf_text(plan.get("additional") or "None specified"))],
    ]
    glance_table = Table(glance_rows, colWidths=[4.2*cm, 10.8*cm])
    glance_table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (0,-1), panel),
        ("BOX", (0,0), (-1,-1), 0.6, border),
        ("INNERGRID", (0,0), (-1,-1), 0.3, border),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("TOPPADDING", (0,0), (-1,-1), 7),
        ("BOTTOMPADDING", (0,0), (-1,-1), 7),
        ("LEFTPADDING", (0,0), (-1,-1), 8),
        ("RIGHTPADDING", (0,0), (-1,-1), 8),
    ]))
    story.append(glance_table)

    # 02. Transport
    story.append(Paragraph("02  •  ✈  TRANSPORT & ROUTES", styles["TMSection"]))
    flight_text = clean_pdf_text(plan.get("flight_results", ""))
    route_box = Table(
        [[P("ROUTE", "TMLabel"), P(f"{origin}  ✈  {destination}", "TMValue")]],
        colWidths=[3.0*cm, 12.0*cm],
    )
    route_box.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (0,0), panel),
        ("BOX", (0,0), (-1,-1), 0.6, border),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING", (0,0), (-1,-1), 8),
        ("BOTTOMPADDING", (0,0), (-1,-1), 8),
        ("LEFTPADDING", (0,0), (-1,-1), 8),
        ("RIGHTPADDING", (0,0), (-1,-1), 8),
    ]))
    story.append(route_box)

    if flight_text:
        story.append(Spacer(1, 6))
        story.append(P("TRANSIT GUIDANCE", "TMLabel"))
        paragraphs = [x.strip() for x in re.split(r"\n\s*\n|(?=\d+\.\s+)|(?=•\s+)|(?=-\s+)", flight_text) if x.strip()]
        for chunk in paragraphs:
            story.append(P(chunk, "TMBody"))
    else:
        story.append(P("No transport guidance was returned.", "TMSmall"))

    # 03. Accommodation
    story.append(Paragraph("03  •  ⌂  STAYS & ACCOMMODATION", styles["TMSection"]))
    parsed_hotels = try_parse_structured(plan.get("hotel_results", ""))
    hotel_entries = (
        parsed_hotels.get("results")
        if isinstance(parsed_hotels, dict) and isinstance(parsed_hotels.get("results"), list)
        else parsed_hotels if isinstance(parsed_hotels, list) else []
    )
    if hotel_entries:
        hrows = [[P("OPTION", "TMLabel"), P("AREA / LOCATION", "TMLabel"), P("WHY IT FITS", "TMLabel")]]
        for item in hotel_entries[:5]:
            if not isinstance(item, dict):
                continue
            title = clean_pdf_text(item.get("title") or item.get("name") or "Accommodation option")
            area = clean_pdf_text(item.get("area") or item.get("location") or "—")
            why = clean_pdf_text(item.get("content") or item.get("snippet") or item.get("description") or "")
            hrows.append([P(title, "TMValue"), P(area), P(why[:550])])
        ht = Table(hrows, colWidths=[4.1*cm, 3.3*cm, 7.6*cm], repeatRows=1)
        ht.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), ink),
            ("TEXTCOLOR", (0,0), (-1,0), white),
            ("BOX", (0,0), (-1,-1), 0.6, border),
            ("INNERGRID", (0,0), (-1,-1), 0.3, border),
            ("VALIGN", (0,0), (-1,-1), "TOP"),
            ("TOPPADDING", (0,0), (-1,-1), 7),
            ("BOTTOMPADDING", (0,0), (-1,-1), 7),
            ("LEFTPADDING", (0,0), (-1,-1), 6),
            ("RIGHTPADDING", (0,0), (-1,-1), 6),
        ]))
        story.append(ht)
    else:
        hotel_text = clean_pdf_text(plan.get("hotel_results", ""))
        story.append(P(hotel_text or "No accommodation recommendations were returned.", "TMSmall"))

    # 04. Weather
    story.append(Paragraph("04  •  ☀  WEATHER & PACKING", styles["TMSection"]))
    current, forecast = parse_weather_results(plan.get("weather_results", ""))
    if isinstance(current, dict) and "temperature_c" in current:
        weather_top = Table([
            [P("TEMPERATURE", "TMLabel"), P("FEELS LIKE", "TMLabel"), P("HUMIDITY", "TMLabel"), P("WIND", "TMLabel")],
            [
                P(f"{current.get('temperature_c','—')} °C", "TMValue"),
                P(f"{current.get('feels_like_c','—')} °C", "TMValue"),
                P(f"{current.get('humidity','—')} %", "TMValue"),
                P(f"{current.get('wind_speed','—')} m/s", "TMValue"),
            ],
        ], colWidths=[3.75*cm]*4)
        weather_top.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,-1), panel),
            ("BOX", (0,0), (-1,-1), 0.6, border),
            ("INNERGRID", (0,0), (-1,-1), 0.3, border),
            ("TOPPADDING", (0,0), (-1,-1), 7),
            ("BOTTOMPADDING", (0,0), (-1,-1), 7),
        ]))
        story.append(weather_top)
        if current.get("condition"):
            story.append(P(f"☁ Current conditions: {clean_pdf_text(current.get('condition'))}", "TMSmall"))

    packing_hint = "Pack comfortable walking shoes, weather-appropriate breathable layers, a reusable bottle, and compact rain protection."
    story.append(P(f"✓ Recommended Packing Cue: {packing_hint}", "TMSmall"))

    if isinstance(forecast, list) and forecast:
        story.append(Spacer(1, 2))
        frows = [[P("DATE / TIME", "TMLabel"), P("TEMP", "TMLabel"), P("CONDITION", "TMLabel")]]
        for item in forecast[:5]:
            if isinstance(item, dict):
                frows.append([
                    P(item.get("datetime", "—")),
                    P(f"{item.get('temperature','—')} °C"),
                    P(clean_pdf_text(item.get("weather", "—")).capitalize()),
                ])
        forecast_table = Table(frows, colWidths=[6.0*cm, 3.0*cm, 6.0*cm], repeatRows=1)
        forecast_table.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), panel),
            ("BOX", (0,0), (-1,-1), 0.6, border),
            ("INNERGRID", (0,0), (-1,-1), 0.3, border),
            ("TOPPADDING", (0,0), (-1,-1), 5),
            ("BOTTOMPADDING", (0,0), (-1,-1), 5),
            ("VALIGN", (0,0), (-1,-1), "TOP"),
        ]))
        story.append(forecast_table)

    # 05. Itinerary
    story.append(PageBreak())
    story.append(Paragraph("05  •  🗓  CHRONOLOGICAL ITINERARY", styles["TMSection"]))
    story.append(P(
        "Arranged chronologically for clear daily pacing. "
        "Timings represent optimized sequence estimates.",
        "TMSmall",
    ))

    if days:
        for index, (title, body) in enumerate(days.items(), start=1):
            day_title = _safe_day_title(title, index)
            date_label = _day_date_label(plan, index)
            activities = _extract_day_activities(body) or [("", "No activities were returned for this day.", "")]

            header = Table([[
                P(f"DAY {index:02d}", "TMLabel"),
                P(date_label, "TMSmall"),
            ]], colWidths=[2.4*cm, 12.6*cm])
            header.setStyle(TableStyle([
                ("BACKGROUND", (0,0), (-1,-1), pale),
                ("BOX", (0,0), (-1,-1), 0.6, colors.HexColor("#B5E2CA")),
                ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
                ("TOPPADDING", (0,0), (-1,-1), 6),
                ("BOTTOMPADDING", (0,0), (-1,-1), 6),
                ("LEFTPADDING", (0,0), (-1,-1), 8),
                ("RIGHTPADDING", (0,0), (-1,-1), 8),
            ]))

            timeline_rows = []
            for time_value, activity, detail in activities[:12]:
                timeline_rows.append([
                    P(time_value or "Flexible", "TMLabel"),
                    P(activity, "TMValue"),
                    P(detail or "", "TMSmall"),
                ])

            day_title_p = P(day_title, "TMSub")
            timeline = Table(
                timeline_rows,
                colWidths=[2.3*cm, 5.0*cm, 7.7*cm],
                repeatRows=0,
                splitByRow=1,
            )
            timeline.setStyle(TableStyle([
                ("BOX", (0,0), (-1,-1), 0.6, border),
                ("INNERGRID", (0,0), (-1,-1), 0.3, border),
                ("BACKGROUND", (0,0), (0,-1), panel),
                ("VALIGN", (0,0), (-1,-1), "TOP"),
                ("TOPPADDING", (0,0), (-1,-1), 7),
                ("BOTTOMPADDING", (0,0), (-1,-1), 7),
                ("LEFTPADDING", (0,0), (-1,-1), 7),
                ("RIGHTPADDING", (0,0), (-1,-1), 7),
            ]))

            day_block = Table([
                [day_title_p],
                [header],
                [timeline],
            ], colWidths=[15*cm])
            day_block.setStyle(TableStyle([
                ("BOX", (0,0), (-1,-1), 0.7, border),
                ("BACKGROUND", (0,0), (-1,0), white),
                ("LEFTPADDING", (0,0), (-1,0), 9),
                ("RIGHTPADDING", (0,0), (-1,0), 9),
                ("TOPPADDING", (0,0), (-1,0), 8),
                ("BOTTOMPADDING", (0,0), (-1,0), 5),
                ("LEFTPADDING", (0,1), (-1,-1), 0),
                ("RIGHTPADDING", (0,1), (-1,-1), 0),
                ("TOPPADDING", (0,1), (-1,-1), 0),
                ("BOTTOMPADDING", (0,1), (-1,-1), 0),
            ]))
            story.extend([KeepTogether(day_block), Spacer(1, 9)])
    else:
        story.append(P(clean_pdf_text(plan.get("itinerary", "")) or "No itinerary was returned.", "TMBody"))

    # 06. Budget Snapshot
    story.append(Paragraph("06  •  ₹  BUDGET BREAKDOWN", styles["TMSection"]))
    story.append(P("Planning estimates only. Verify live prices and availability before booking.", "TMSmall"))

    budget_rows = [[P("CATEGORY", "TMLabel"), P("ESTIMATE", "TMLabel"), P("DETAIL", "TMLabel")]]
    if budget:
        budget_rows.append([P("Total Target Budget", "TMValue"), P(budget_text, "TMValue"), P("User-specified ceiling")])

    for title, body in budget_sections.items():
        clean_body = clean_pdf_text(body)
        amount_matches = re.findall(r"(?:₹|INR\s*)?[\d][\d,]*(?:\.\d+)?", clean_body)
        amount = amount_matches[0] if amount_matches else "Estimate"
        label = clean_pdf_text(title).replace("Budget", "").replace("budget", "").strip(" :-") or "Estimated Expenses"
        budget_rows.append([P(label, "TMValue"), P(amount, "TMValue"), P(clean_body[:300])])

    if len(budget_rows) == 1:
        budget_rows.append([P("Category breakdown"), P("—"), P("Refer to itinerary notes")])

    bt = Table(budget_rows, colWidths=[5.1*cm, 3.5*cm, 6.4*cm], repeatRows=1)
    bt.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), ink),
        ("TEXTCOLOR", (0,0), (-1,0), white),
        ("BOX", (0,0), (-1,-1), 0.6, border),
        ("INNERGRID", (0,0), (-1,-1), 0.3, border),
        ("BACKGROUND", (0,1), (-1,-1), white),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("ALIGN", (1,1), (1,-1), "RIGHT"),
        ("TOPPADDING", (0,0), (-1,-1), 7),
        ("BOTTOMPADDING", (0,0), (-1,-1), 7),
        ("LEFTPADDING", (0,0), (-1,-1), 7),
        ("RIGHTPADDING", (0,0), (-1,-1), 7),
    ]))
    story.append(bt)

    # 07. Travel Notes
    story.append(Paragraph("07  •  ✓  TRAVEL NOTES & CUES", styles["TMSection"]))
    if tips:
        for title, body in tips.items():
            tip_table = Table(
                [[P(f"✓  {clean_pdf_text(title)}", "TMValue")],
                 [P(clean_pdf_text(body))]],
                colWidths=[15*cm],
            )
            tip_table.setStyle(TableStyle([
                ("BACKGROUND", (0,0), (-1,0), panel),
                ("BOX", (0,0), (-1,-1), 0.5, border),
                ("LEFTPADDING", (0,0), (-1,-1), 8),
                ("RIGHTPADDING", (0,0), (-1,-1), 8),
                ("TOPPADDING", (0,0), (-1,-1), 7),
                ("BOTTOMPADDING", (0,0), (-1,-1), 7),
            ]))
            story.extend([tip_table, Spacer(1, 6)])
    else:
        story.append(P(
            "Verify passport and visa requirements, reconfirm flight check-in and hotel dates, "
            "and maintain a flexible buffer for local transit.",
            "TMBody",
        ))

    story.extend([
        Spacer(1, 12),
        HRFlowable(width="100%", thickness=0.8, color=border, spaceBefore=3, spaceAfter=7),
        P(
            f"{APP_NAME}  •  Personalized Travel Blueprint  •  {CREATOR_CREDIT}  •  Generated {datetime.now().strftime('%d %b %Y')}.",
            "TMSmall",
        ),
    ])

    doc.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)
    return buffer.getvalue()


# ─────────────────────────────────────────────────────────────────────────
# Main Application Entry Point
# ─────────────────────────────────────────────────────────────────────────
def main() -> None:
    st.set_page_config(
        page_title=f"{APP_NAME} — {APP_TAGLINE}",
        page_icon="🧭",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    inject_css()
    init_session_state()
    render_sidebar()
    render_topbar()

    if (
        st.session_state.get("generation_error")
        and st.session_state["plan"] is None
    ):
        st.error(
            "TravelMind could not complete this journey. "
            "Please check your travel services and try again."
        )

    current_view = st.session_state.get("view", "home")

    if current_view == "home":
        render_input_screen()
        return

    if current_view == "processing":
        render_processing_screen()
        return

    if current_view == "approval" and st.session_state.get("plan"):
        render_approval_screen(st.session_state["plan"])
        return

    if current_view == "result" and st.session_state.get("plan"):
        render_result_screen(st.session_state["plan"])
        return

    render_input_screen()


if __name__ == "__main__":
    main()
