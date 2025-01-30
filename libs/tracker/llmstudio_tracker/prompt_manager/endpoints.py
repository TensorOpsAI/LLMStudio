from fastapi import APIRouter, Depends
from llmstudio_tracker.database import engine, get_db
from llmstudio_tracker.prompt_manager import crud, models, schemas
from sqlalchemy.orm import Session

models.Base.metadata.create_all(bind=engine)


class PromptsRoutes:
    def __init__(self, router: APIRouter):
        self.router = router
        self.define_routes()

    def define_routes(self):
        self.router.post(
            "/add/prompt",
            response_model=schemas.PromptDefault,
        )(self.add_prompt)

        self.router.get("/get/prompt", response_model=schemas.PromptDefault)(
            self.get_prompt
        )

        self.router.patch("/update/prompt", response_model=schemas.PromptDefault)(
            self.update_prompt
        )

        self.router.delete("/delete/prompt")(self.delete_prompt)

    async def add_prompt(
        self, prompt: schemas.PromptDefault, db: Session = Depends(get_db)
    ):
        return crud.add_prompt(db=db, prompt=prompt)

    async def update_prompt(
        self, prompt: schemas.PromptDefault, db: Session = Depends(get_db)
    ):
        return crud.update_prompt(db, prompt)

    async def get_prompt(
        self,
        prompt_info: schemas.PromptInfo,
        db: Session = Depends(get_db),
    ):
        return crud.get_prompt(
            db,
            prompt_id=prompt_info.prompt_id,
            name=prompt_info.name,
            model=prompt_info.model,
            provider=prompt_info.provider,
        )

    async def delete_prompt(
        self, prompt: schemas.PromptDefault, db: Session = Depends(get_db)
    ):
        return crud.delete_prompt(db, prompt)
