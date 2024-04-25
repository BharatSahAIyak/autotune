import os

import dj_database_url
import psycopg2
from django.core.management import call_command
from django.core.management.base import BaseCommand
from psycopg2 import sql


class Command(BaseCommand):
    help = "Ensure the database exists, create it if necessary, and apply migrations."

    def handle(self, *args, **kwargs):
        # Parse the database URL from environment variables
        db_config = dj_database_url.parse(os.getenv("DATABASE_URL"))
        db_name = db_config["NAME"]

        # Connect to the default 'postgres' database to check/create the target database
        connection = psycopg2.connect(
            dbname="postgres",
            user=db_config["USER"],
            password=db_config["PASSWORD"],
            host=db_config["HOST"],
            port=db_config["PORT"],
        )
        connection.autocommit = True
        cursor = connection.cursor()

        # Check if the database exists
        cursor.execute(
            sql.SQL("SELECT 1 FROM pg_database WHERE datname = %s"), [db_name]
        )
        if not cursor.fetchone():
            # Create the database if it doesn't exist
            cursor.execute(
                sql.SQL("CREATE DATABASE {}").format(sql.Identifier(db_name))
            )
            self.stdout.write(self.style.SUCCESS(f"Created database '{db_name}'"))
        else:
            self.stdout.write(
                self.style.SUCCESS(f"Database '{db_name}' already exists")
            )

        cursor.close()
        connection.close()

        # Now apply migrations to the database
        self.stdout.write("Applying migrations to the database...")
        call_command("migrate")
        self.stdout.write(self.style.SUCCESS("Migrations applied successfully."))
