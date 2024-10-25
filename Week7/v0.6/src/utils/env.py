from dataclasses import dataclass
from dotenv import load_dotenv
import os

load_dotenv()


@dataclass
class Env:
    fred_api_key: str = os.getenv("FRED_API_KEY")


env = Env()
