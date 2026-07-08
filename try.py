from app.services.job_role.model_loader import _load_model, _load_skill_map, _load_tfidf, _load_encoder
from app.services.job_role.predictor import _parse_skills, _get_scores, predict_job_role

# print(_load_model().summary())
# print(_load_encoder())
# print(_load_skill_map())
# print(_load_tfidf())

skills = ["React", "Next", "Tailwindcss", "Vue", "Angular"]
# print(_get_scores(skills))
print(predict_job_role(skills))