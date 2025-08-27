"""
Authentication endpoints for ESCAI Framework API.
"""

from datetime import datetime, timedelta
from typing import Dict, Any, List

from fastapi import APIRouter, HTTPException, status, Depends, Request
from fastapi.security import HTTPBearer
from pydantic import BaseModel
from slowapi import Limiter
from slowapi.util import get_remote_address

from .auth import (
    AuthManager, User, UserLogin, Token, UserCreate, 
    auth_manager, get_current_user, require_admin,
    ACCESS_TOKEN_EXPIRE_MINUTES
)
from ..utils.logging import get_logger

logger = get_logger(__name__)
limiter = Limiter(key_func=get_remote_address)

# Router
auth_router = APIRouter()

# Request/Response models
class LoginResponse(BaseModel):
    """Login response model."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    user: Dict[str, Any]

class RefreshTokenRequest(BaseModel):
    """Refresh token request."""
    refresh_token: str

class ChangePasswordRequest(BaseModel):
    """Change password request."""
    current_password: str
    new_password: str

@auth_router.post("/login", response_model=LoginResponse)
@limiter.limit("5/minute")
async def login(request: Request, user_credentials: UserLogin) -> LoginResponse:
    """Authenticate user and return tokens."""
    try:
        # Authenticate user
        user = auth_manager.authenticate_user(
            user_credentials.username, 
            user_credentials.password
        )
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Create tokens
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = auth_manager.create_access_token(
            data={
                "sub": user["user_id"],
                "username": user["username"],
                "roles": [role.value for role in user["roles"]]
            },
            expires_delta=access_token_expires
        )
        
        refresh_token = auth_manager.create_refresh_token(
            data={
                "sub": user["user_id"],
                "username": user["username"]
            }
        )
        
        logger.info(f"User {user['username']} logged in successfully")
        
        return LoginResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            user={
                "user_id": user["user_id"],
                "username": user["username"],
                "email": user["email"],
                "roles": [role.value for role in user["roles"]],
                "last_login": user.get("last_login").isoformat() if user.get("last_login") and hasattr(user.get("last_login"), "isoformat") else None
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed"
        )

@auth_router.post("/refresh", response_model=Token)
@limiter.limit("10/minute")
async def refresh_token(request: Request, refresh_request: RefreshTokenRequest) -> Token:
    """Refresh access token using refresh token."""
    try:
        new_token = auth_manager.refresh_access_token(refresh_request.refresh_token)
        
        if not new_token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        return new_token
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Token refresh failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Token refresh failed"
        )

@auth_router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
@limiter.limit("10/minute")
async def logout(
    request: Request,
    refresh_request: RefreshTokenRequest,
    current_user: User = Depends(get_current_user)
) -> None:
    """Logout user and revoke refresh token."""
    try:
        # Revoke refresh token
        auth_manager.revoke_refresh_token(refresh_request.refresh_token)
        
        logger.info(f"User {current_user.username} logged out successfully")
        
    except Exception as e:
        logger.error(f"Logout failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Logout failed"
        )

@auth_router.get("/me", response_model=Dict[str, Any])
@limiter.limit("30/minute")
async def get_current_user_info(
    request: Request,
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get current user information."""
    return {
        "user_id": current_user.user_id,
        "username": current_user.username,
        "email": current_user.email,
        "roles": [role.value for role in current_user.roles],
        "is_active": current_user.is_active,
        "created_at": current_user.created_at.isoformat() if hasattr(current_user.created_at, "isoformat") else str(current_user.created_at),
        "last_login": current_user.last_login.isoformat() if current_user.last_login and hasattr(current_user.last_login, "isoformat") else None
    }

@auth_router.post("/change-password", status_code=status.HTTP_204_NO_CONTENT)
@limiter.limit("3/minute")
async def change_password(
    request: Request,
    password_request: ChangePasswordRequest,
    current_user: User = Depends(get_current_user)
) -> None:
    """Change user password."""
    try:
        # Get user from database
        user_dict = auth_manager.users_db.get(current_user.username)
        if not user_dict:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Verify current password
        if not auth_manager.verify_password(
            password_request.current_password, 
            user_dict["hashed_password"]
        ):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Current password is incorrect"
            )
        
        # Update password
        user_dict["hashed_password"] = auth_manager.get_password_hash(
            password_request.new_password
        )
        
        logger.info(f"Password changed for user {current_user.username}")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Password change failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Password change failed"
        )

@auth_router.post("/users", response_model=Dict[str, Any])
@limiter.limit("5/minute")
async def create_user(
    request: Request,
    user_data: UserCreate,
    current_user: User = Depends(require_admin())
) -> Dict[str, Any]:
    """Create new user (admin only)."""
    try:
        # Check if user already exists
        if user_data.username in auth_manager.users_db:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Username already exists"
            )
        
        # Create user
        user_id = f"user-{len(auth_manager.users_db) + 1:03d}"
        new_user = {
            "user_id": user_id,
            "email": user_data.email,
            "username": user_data.username,
            "hashed_password": auth_manager.get_password_hash(user_data.password),
            "roles": user_data.roles,
            "is_active": True,
            "created_at": datetime.utcnow()
        }
        
        auth_manager.users_db[user_data.username] = new_user
        
        logger.info(f"User {user_data.username} created by {current_user.username}")
        
        return {
            "user_id": user_id,
            "username": user_data.username,
            "email": user_data.email,
            "roles": [role.value for role in user_data.roles],
            "is_active": True,
            "created_at": new_user["created_at"].isoformat() if new_user["created_at"] and hasattr(new_user["created_at"], "isoformat") else str(new_user["created_at"])
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"User creation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="User creation failed"
        )

@auth_router.get("/users", response_model=List[Dict[str, Any]])
@limiter.limit("10/minute")
async def list_users(
    request: Request,
    current_user: User = Depends(require_admin())
) -> List[Dict[str, Any]]:
    """List all users (admin only)."""
    try:
        users = []
        for username, user_dict in auth_manager.users_db.items():
            users.append({
                "user_id": user_dict["user_id"],
                "username": user_dict["username"],
                "email": user_dict["email"],
                "roles": [role.value for role in user_dict["roles"]],
                "is_active": user_dict["is_active"],
                "created_at": user_dict["created_at"].isoformat() if user_dict["created_at"] and hasattr(user_dict["created_at"], "isoformat") else str(user_dict["created_at"]),
                "last_login": user_dict.get("last_login").isoformat() if user_dict.get("last_login") and hasattr(user_dict.get("last_login"), "isoformat") else None
            })
        
        return users
        
    except Exception as e:
        logger.error(f"Failed to list users: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list users"
        )

@auth_router.put("/users/{username}/status", status_code=status.HTTP_204_NO_CONTENT)
@limiter.limit("5/minute")
async def update_user_status(
    request: Request,
    username: str,
    is_active: bool,
    current_user: User = Depends(require_admin())
) -> None:
    """Update user active status (admin only)."""
    try:
        if username not in auth_manager.users_db:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        auth_manager.users_db[username]["is_active"] = is_active
        
        logger.info(f"User {username} status updated to {'active' if is_active else 'inactive'} by {current_user.username}")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update user status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update user status"
        )