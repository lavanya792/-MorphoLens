from .database import Base, engine
from . import models


def initialize_database():

    Base.metadata.create_all(bind=engine)

    print("\nMorphoLens database initialized!")

    print("\nCreated tables:")

    for table in Base.metadata.sorted_tables:
        print("-", table.name)


if __name__ == "__main__":
    initialize_database()