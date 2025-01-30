from llmstudio_tracker.prompt_manager import models, schemas
from sqlalchemy.orm import Session


def get_prompt_by_name_model_provider(
    db: Session, name: str, model: str, provider: str
):
    return (
        db.query(models.PromptDefault)
        .filter(
            models.PromptDefault.name == name,
            models.PromptDefault.model == model,
            models.PromptDefault.provider == provider,
            models.PromptDefault.is_active == True,
        )
        .order_by(models.PromptDefault.version.desc())
        .first()
    )


def get_prompt_by_id(db: Session, prompt_id: str):
    return (
        db.query(models.PromptDefault)
        .filter(models.PromptDefault.prompt_id == prompt_id)
        .first()
    )


def get_prompt(
    db: Session,
    prompt_id: str = None,
    name: str = None,
    model: str = None,
    provider: str = None,
):
    prompt = (
        get_prompt_by_id(db, prompt_id)
        if prompt_id
        else get_prompt_by_name_model_provider(db, name, model, provider)
    )

    if not prompt:
        return schemas.PromptDefault()

    return prompt


def add_prompt(db: Session, prompt: schemas.PromptDefault):

    prompt_created = models.PromptDefault.create_with_incremental_version(
        db,
        config=prompt.config,
        prompt=prompt.prompt,
        is_active=prompt.is_active,
        name=prompt.name,
        label=prompt.label,
        model=prompt.model,
        provider=prompt.provider,
    )
    db.add(prompt_created)
    db.commit()
    db.refresh(prompt_created)
    return prompt_created


def update_prompt(db: Session, prompt: schemas.PromptDefault):
    if prompt.prompt_id:
        existing_prompt = get_prompt_by_id(db, prompt.prompt_id)
    else:
        existing_prompt = get_prompt_by_name_model_provider(
            db, prompt.name, prompt.model, prompt.provider
        )

    existing_prompt.config = prompt.config
    existing_prompt.prompt = prompt.prompt
    existing_prompt.is_active = prompt.is_active
    existing_prompt.name = prompt.name
    existing_prompt.model = prompt.model
    existing_prompt.provider = prompt.provider
    existing_prompt.version = prompt.version
    existing_prompt.label = prompt.label

    db.commit()
    db.refresh(existing_prompt)
    return existing_prompt


def delete_prompt(db: Session, prompt: schemas.PromptDefault):
    if prompt.prompt_id:
        existing_prompt = get_prompt_by_id(db, prompt.prompt_id)
    else:
        existing_prompt = get_prompt_by_name_model_provider(
            db, prompt.name, prompt.model, prompt.provider
        )

    db.delete(existing_prompt)
    db.commit()
