import logging

import uvicorn

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    uvicorn.run(
        "production_rag_agent.app:factory",
        factory=True,
        host="127.0.0.1",
        port=8000,
        workers=1,
        access_log=False,
    )
