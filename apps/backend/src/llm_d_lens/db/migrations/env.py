from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

import llm_d_lens.db.models  # noqa: F401

# Import the models package so Base.metadata is fully populated before
# autogenerate compares it against the live database (see
# docs/design/database-design.md#data-access-architecture section 8, step 2).
# `llm_d_lens.db.models.__init__` is the single registration list.
from llm_d_lens.db.base import Base
from llm_d_lens.db.settings import DatabaseSettings, resolve_database_url

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name, disable_existing_loggers=False)

# add your model's MetaData object here
# for 'autogenerate' support
target_metadata = Base.metadata

# Prefer the same DATABASE_URL resolution used at application runtime (see
# llm_d_lens/db/settings.py) over a static alembic.ini value, unless the
# caller explicitly passed -x sqlalchemy.url=... on the CLI.
if not config.get_main_option("sqlalchemy.url"):
    # set_main_option() runs the value through ConfigParser, which treats a
    # raw "%" as the start of a "%(name)s" interpolation token -- escape any
    # literal "%" (e.g. from a URL-encoded password like "%2B") as "%%" so
    # generated passwords/URLs containing them don't break config parsing.
    resolved_url = resolve_database_url(DatabaseSettings.from_environment())
    config.set_main_option("sqlalchemy.url", resolved_url.replace("%", "%%"))

# other values from the config, defined by the needs of env.py,
# can be acquired:
# my_important_option = config.get_main_option("my_important_option")
# ... etc.


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.

    """
    supplied_connection = config.attributes.get("connection")
    if supplied_connection is not None:
        context.configure(connection=supplied_connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()
        return

    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
