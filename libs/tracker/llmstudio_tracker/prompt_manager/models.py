import uuid
from datetime import datetime, timezone

from llmstudio_tracker.config import DB_TYPE
from llmstudio_tracker.database import Base
from llmstudio_tracker.db_utils import JSONEncodedDict
from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Integer,
    String,
    UniqueConstraint,
    event,
    func,
)
from sqlalchemy.orm import Session


class PromptDefault(Base):
    __tablename__ = "prompts"

    if DB_TYPE == "bigquery":
        prompt_id = Column(
            String,
            primary_key=True,
            default=lambda: str(uuid.uuid4()),
        )
        config = Column(JSONEncodedDict, nullable=True)
    else:
        prompt_id = Column(
            String, primary_key=True, default=lambda: str(uuid.uuid4())
        )  # Generate UUID as a string
        config = Column(JSON, nullable=True)

    prompt = Column(String)
    is_active = Column(Boolean, default=False)
    name = Column(String, nullable=False)
    model = Column(String, nullable=False)
    provider = Column(String, nullable=False)
    version = Column(Integer, nullable=False)
    label = Column(String)
    updated_at = Column(
        DateTime(timezone=True),
        onupdate=lambda: datetime.now(timezone.utc),
        default=lambda: datetime.now(timezone.utc),
    )
    created_at = Column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    __table_args__ = (
        UniqueConstraint(
            "name", "provider", "model", "version", name="uq_prompt_version"
        ),
    )

    @staticmethod
    def get_next_version(session, name, model, provider):
        """
        Get the next version number for a combination of name, model, and provider.
        """
        max_version = (
            session.query(func.max(PromptDefault.version))
            .filter_by(name=name, model=model, provider=provider)
            .scalar()
        )
        return (max_version or 0) + 1

    @classmethod
    def create_with_incremental_version(cls, session, **kwargs):
        """
        Create a new PromptDefault entry with an incremental version.
        """
        name = kwargs.get("name")
        model = kwargs.get("model")
        provider = kwargs.get("provider")
        if not all([name, model, provider]):
            raise ValueError("name, model, and provider must be provided")

        kwargs["version"] = cls.get_next_version(session, name, model, provider)

        instance = cls(**kwargs)
        session.add(instance)
        return instance

    @event.listens_for(Session, "before_flush")
    def ensure_single_active_prompt(session, flush_context, instances):
        """
        Ensures only one PromptDefault entry per (name, model, provider) can have is_active=True.
        If a new entry is set as is_active=True, deactivate others in the same group.
        """
        for instance in session.new.union(session.dirty):
            if isinstance(instance, PromptDefault) and instance.is_active:
                session.query(PromptDefault).filter(
                    PromptDefault.name == instance.name,
                    PromptDefault.model == instance.model,
                    PromptDefault.provider == instance.provider,
                    PromptDefault.is_active == True,
                    PromptDefault.prompt_id != instance.prompt_id,
                ).update({"is_active": False}, synchronize_session="fetch")
