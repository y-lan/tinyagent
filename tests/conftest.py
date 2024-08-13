from dotenv import load_dotenv
import pytest


@pytest.fixture(scope="session", autouse=True)
def load_env():
    load_dotenv()
