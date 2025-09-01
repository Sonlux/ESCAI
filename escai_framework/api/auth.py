"""
Authentication and authorization for ESCAI Framework API.
"""

import secrets
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any
from enum import Enum

import jwt
from passlib.context import CryptContext
from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr

from ..utils.logging import get_logger

logger = get_logger(__name__)

# Configuration
SECRET_KEY = secrets.token_urlsafe(32)  # In production, use environment variable
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

class UserRole(str, Enum):
    """User roles for RBAC."""
    ADMIN = "admin"
    RESEARCHER = "researcher"
    DEVELOPER = "developer"
    VIEWER = "viewer"

class User(BaseModel):
    """User model."""
    user_id: str
    email: EmailStr
    username: str
    roles: List[UserRole]
    is_active: bool = True
    created_at: datetime
    last_login: Optional[datetime] = None

class UserCreate(BaseModel):
    """User creation model."""
    email: EmailStr
    username: str
    password: str
    roles: List[UserRole] = [UserRole.VIEWER]

class UserLogin(BaseModel):
    """User login model."""
    username: str
    password: str

class Token(BaseModel):
    """Token response model."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int

class TokenData(BaseModel):
    """Token data model."""
    user_id: Optional[str] = None
    username: Optional[str] = None
    roles: List[str] = []

class AuthManager:
    """Authentication and authorization manager."""
    
    def __init__(self):
        self.users_db: Dict[str, Dict] = {}  # In production, use proper database
        self.refresh_tokens: Dict[str, str] = {}  # In production, use Redis
        
        # Create default admin user
        self._create_default_users()
    
    def _create_default_users(self) -> None:
        """Create default users for development."""
        admin_user = {
            "user_id": "admin-001",
            "email": "admin@escai.dev",
            "username": "admin",
            "hashed_password": self.get_password_hash("admin123"),
            "roles": [UserRole.ADMIN],
            "is_active": True,
            "created_at": datetime.utcnow()
        }
        
        researcher_user = {
            "user_id": "researcher-001",
            "email": "researcher@escai.dev",
            "username": "researcher",
            "hashed_password": self.get_password_hash("research123"),
            "roles": [UserRole.RESEARCHER],
            "is_active": True,
            "created_at": datetime.utcnow()
        }
        
        self.users_db["admin"] = admin_user
        self.users_db["researcher"] = researcher_user
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify password against hash."""
        return pwd_context.verify(plain_password, hashed_password)
    
    def get_password_hash(self, password: str) -> str:
        """Hash password."""
        return pwd_context.hash(password)
    
    def authenticate_user(self, username: str, password: str) -> Optional[Dict]:
        """Authenticate user credentials."""
        user = self.users_db.get(username)
        if not user:
            return None
        if not self.verify_password(password, user["hashed_password"]):
            return None
        if not user["is_active"]:
            return None
        
        # Update last login
        user["last_login"] = datetime.utcnow()
        return user
    
    def create_access_token(self, data: Dict, expires_delta: Optional[timedelta] = None) -> str:
        """Create JWT access token."""
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        
        to_encode.update({"exp": expire, "type": "access"})
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        return encoded_jwt
    
    def create_refresh_token(self, data: Dict) -> str:
        """Create JWT refresh token."""
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
        to_encode.update({"exp": expire, "type": "refresh"})
        
        refresh_token = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        
        # Store refresh token
        user_id = data.get("sub")
        if user_id:
            self.refresh_tokens[refresh_token] = user_id
        
        return refresh_token
    
    def verify_token(self, token: str, token_type: str = "access") -> Optional[TokenData]:
        """Verify JWT token."""
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            
            # Check token type
            if payload.get("type") != token_type:
                return None
            
            user_id: str = payload.get("sub")
            username: str = payload.get("username")
            roles: List[str] = payload.get("roles", [])
            
            if user_id is None or username is None:
                return None
            
            return TokenData(user_id=user_id, username=username, roles=roles)
        
        except jwt.PyJWTError:
            return None
    
    def refresh_access_token(self, refresh_token: str) -> Optional[Token]:
        """Refresh access token using refresh token."""
        # Verify refresh token
        token_data = self.verify_token(refresh_token, "refresh")
        if not token_data:
            return None
        
        # Check if refresh token is stored
        if refresh_token not in self.refresh_tokens:
            return None
        
        # Get user data
        if token_data.username is None:
            return None
        user = self.users_db.get(token_data.username)
        if not user or not user["is_active"]:
            return None
        
        # Create new tokens
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = self.create_access_token(
            data={
                "sub": user["user_id"],
                "username": user["username"],
                "roles": [role.value for role in user["roles"]]
            },
            expires_delta=access_token_expires
        )
        
        new_refresh_token = self.create_refresh_token(
            data={
                "sub": user["user_id"],
                "username": user["username"]
            }
        )
        
        # Remove old refresh token
        del self.refresh_tokens[refresh_token]
        
        return Token(
            access_token=access_token,
            refresh_token=new_refresh_token,
            expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60
        )
    
    def revoke_refresh_token(self, refresh_token: str) -> bool:
        """Revoke refresh token."""
        if refresh_token in self.refresh_tokens:
            del self.refresh_tokens[refresh_token]
            return True
        return False
    
    def has_permission(self, user_roles: List[Union[str, UserRole]], required_roles: List[UserRole]) -> bool:
        """Check if user has required permissions."""
        # Convert user_roles to strings for comparison
        user_role_strings = [role.value if isinstance(role, UserRole) else role for role in user_roles]
        required_role_strings = [role.value for role in required_roles]
        
        if UserRole.ADMIN.value in user_role_strings:
            return True
        
        return any(role in user_role_strings for role in required_role_strings)

# Global auth manager instance
auth_manager = AuthManager()

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> User:
    """Get current authenticated user."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    token_data = auth_manager.verify_token(credentials.credentials)
    if token_data is None or token_data.username is None:
        raise credentials_exception
    
    user_dict = auth_manager.users_db.get(token_data.username)
    if user_dict is None:
        raise credentials_exception
    
    return User(
        user_id=user_dict["user_id"],
        email=user_dict["email"],
        username=user_dict["username"],
        roles=user_dict["roles"],
        is_active=user_dict["is_active"],
        created_at=user_dict["created_at"],
        last_login=user_dict.get("last_login")
    )

def require_roles(required_roles: List[UserRole]) -> Any:
    """Decorator to require specific roles."""
    def role_checker(current_user: User = Depends(get_current_user)) -> User:
        # Cast to compatible type for has_permission method
        if not auth_manager.has_permission(list(current_user.roles), required_roles):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions"
            )
        return current_user
    
    return role_checker

# Convenience functions for common role requirements
def require_admin() -> Any:
    return require_roles([UserRole.ADMIN])

def require_researcher() -> Any:
    return require_roles([UserRole.RESEARCHER, UserRole.ADMIN])

def require_developer() -> Any:
    return require_roles([UserRole.DEVELOPER, UserRole.ADMIN])

def require_viewer() -> Any:
    return require_roles([UserRole.VIEWER, UserRole.DEVELOPER, UserRole.RESEARCHER, UserRole.ADMIN])