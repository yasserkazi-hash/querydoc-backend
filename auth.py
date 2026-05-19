import os
from fastapi import Depends, HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import jwt
import requests
from dotenv import load_dotenv

load_dotenv()

CLERK_JWKS_URL = os.getenv("CLERK_JWKS_URL")
CLERK_ISSUER = os.getenv("CLERK_JWT_ISSUER")

security = HTTPBearer()
_jwks_cache = {}

def get_jwks():
    if not _jwks_cache:
        response = requests.get(CLERK_JWKS_URL)
        response.raise_for_status()
        _jwks_cache.update(response.json())
    return _jwks_cache

async def get_current_user(credentials: HTTPAuthorizationCredentials = Security(security)):
    token = credentials.credentials
    try:
        unverified_header = jwt.get_unverified_header(token)
        kid = unverified_header["kid"]
        jwks = get_jwks()
        key = next((jwk for jwk in jwks["keys"] if jwk["kid"] == kid), None)
        if not key:
            raise HTTPException(status_code=401, detail="Invalid token: kid not found")
        payload = jwt.decode(
            token,
            key,
            algorithms=["RS256"],
            issuer=CLERK_ISSUER,
        )
        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid token: missing sub")
        return user_id
    except jwt.JWTError as e:
        raise HTTPException(status_code=401, detail=f"Invalid token: {str(e)}")