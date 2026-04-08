import uvicorn
from openenv.core.env_server import create_fastapi_app
from incident_env.models import IncidentEnvAction, IncidentEnvObservation
from incident_env.server.environment import IncidentEnvEnvironment

app = create_fastapi_app(IncidentEnvEnvironment, IncidentEnvAction, IncidentEnvObservation)

def main():
    uvicorn.run("incident_env.server.app:app", host="0.0.0.0", port=8000, reload=False)

if __name__ == "__main__":
    main()
