"""
Role-Based Access Control (RBAC) System

Provides comprehensive role-based access control with fine-grained permissions,
hierarchical roles, and dynamic permission evaluation.
"""

import asyncio
from enum import Enum
from typing import Dict, List, Set, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import redis.asyncio as redis
import json
import logging

logger = logging.getLogger(__name__)


class ResourceType(Enum):
    """Types of resources that can be protected"""
    AGENT = "agent"
    EPISTEMIC_STATE = "epistemic_state"
    BEHAVIORAL_PATTERN = "behavioral_pattern"
    CAUSAL_RELATIONSHIP = "causal_relationship"
    PREDICTION = "prediction"
    MONITORING_SESSION = "monitoring_session"
    SYSTEM_CONFIG = "system_config"
    USER_MANAGEMENT = "user_management"
    AUDIT_LOG = "audit_log"


class Action(Enum):
    """Actions that can be performed on resources"""
    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    EXECUTE = "execute"
    MANAGE = "manage"
    ADMIN = "admin"


@dataclass(frozen=True)
class Permission:
    """Individual permission for a resource and action"""
    resource_type: ResourceType
    action: Action
    resource_id: Optional[str] = None  # Specific resource ID or None for all
    conditions: Optional[Dict[str, Any]] = None
    
    def __str__(self) -> str:
        resource_str = f"{self.resource_type.value}"
        if self.resource_id:
            resource_str += f":{self.resource_id}"
        return f"{resource_str}:{self.action.value}"
    
    def matches(self, resource_type: ResourceType, action: Action, resource_id: str = None) -> bool:
        """Check if this permission matches the requested access"""
        if self.resource_type != resource_type or self.action != action:
            return False
        
        # If permission is for all resources of this type
        if self.resource_id is None:
            return True
        
        # If permission is for specific resource
        return self.resource_id == resource_id


@dataclass
class Role:
    """Role containing multiple permissions"""
    name: str
    description: str
    permissions: Set[Permission] = field(default_factory=set)
    parent_roles: Set[str] = field(default_factory=set)
    is_system_role: bool = False
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def add_permission(self, permission: Permission):
        """Add permission to role"""
        self.permissions.add(permission)
    
    def remove_permission(self, permission: Permission):
        """Remove permission from role"""
        self.permissions.discard(permission)
    
    def has_permission(self, resource_type: ResourceType, action: Action, resource_id: str = None) -> bool:
        """Check if role has specific permission"""
        return any(
            perm.matches(resource_type, action, resource_id)
            for perm in self.permissions
        )


class RBACManager:
    """Role-Based Access Control Manager"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.roles: Dict[str, Role] = {}
        self.user_roles: Dict[str, Set[str]] = {}
        self._initialize_system_roles()
    
    def _initialize_system_roles(self):
        """Initialize default system roles"""
        
        # Super Admin Role
        super_admin = Role(
            name="super_admin",
            description="Full system access",
            is_system_role=True
        )
        super_admin.permissions.update([
            Permission(ResourceType.AGENT, Action.ADMIN),
            Permission(ResourceType.EPISTEMIC_STATE, Action.ADMIN),
            Permission(ResourceType.BEHAVIORAL_PATTERN, Action.ADMIN),
            Permission(ResourceType.CAUSAL_RELATIONSHIP, Action.ADMIN),
            Permission(ResourceType.PREDICTION, Action.ADMIN),
            Permission(ResourceType.MONITORING_SESSION, Action.ADMIN),
            Permission(ResourceType.SYSTEM_CONFIG, Action.ADMIN),
            Permission(ResourceType.USER_MANAGEMENT, Action.ADMIN),
            Permission(ResourceType.AUDIT_LOG, Action.ADMIN),
        ])
        
        # Admin Role
        admin = Role(
            name="admin",
            description="Administrative access to most resources",
            is_system_role=True
        )
        admin.permissions.update([
            Permission(ResourceType.AGENT, Action.MANAGE),
            Permission(ResourceType.EPISTEMIC_STATE, Action.MANAGE),
            Permission(ResourceType.BEHAVIORAL_PATTERN, Action.MANAGE),
            Permission(ResourceType.CAUSAL_RELATIONSHIP, Action.MANAGE),
            Permission(ResourceType.PREDICTION, Action.MANAGE),
            Permission(ResourceType.MONITORING_SESSION, Action.MANAGE),
            Permission(ResourceType.SYSTEM_CONFIG, Action.READ),
            Permission(ResourceType.USER_MANAGEMENT, Action.READ),
            Permission(ResourceType.AUDIT_LOG, Action.READ),
        ])
        
        # Analyst Role
        analyst = Role(
            name="analyst",
            description="Analysis and monitoring access",
            is_system_role=True
        )
        analyst.permissions.update([
            Permission(ResourceType.AGENT, Action.READ),
            Permission(ResourceType.EPISTEMIC_STATE, Action.READ),
            Permission(ResourceType.BEHAVIORAL_PATTERN, Action.READ),
            Permission(ResourceType.CAUSAL_RELATIONSHIP, Action.READ),
            Permission(ResourceType.PREDICTION, Action.READ),
            Permission(ResourceType.MONITORING_SESSION, Action.CREATE),
            Permission(ResourceType.MONITORING_SESSION, Action.READ),
            Permission(ResourceType.MONITORING_SESSION, Action.UPDATE),
        ])
        
        # Monitor Role
        monitor = Role(
            name="monitor",
            description="Basic monitoring access",
            is_system_role=True
        )
        monitor.permissions.update([
            Permission(ResourceType.AGENT, Action.READ),
            Permission(ResourceType.EPISTEMIC_STATE, Action.READ),
            Permission(ResourceType.MONITORING_SESSION, Action.CREATE),
            Permission(ResourceType.MONITORING_SESSION, Action.READ),
        ])
        
        # Viewer Role
        viewer = Role(
            name="viewer",
            description="Read-only access",
            is_system_role=True
        )
        viewer.permissions.update([
            Permission(ResourceType.AGENT, Action.READ),
            Permission(ResourceType.EPISTEMIC_STATE, Action.READ),
            Permission(ResourceType.BEHAVIORAL_PATTERN, Action.READ),
            Permission(ResourceType.CAUSAL_RELATIONSHIP, Action.READ),
            Permission(ResourceType.PREDICTION, Action.READ),
        ])
        
        self.roles.update({
            "super_admin": super_admin,
            "admin": admin,
            "analyst": analyst,
            "monitor": monitor,
            "viewer": viewer
        })
    
    async def create_role(self, role: Role) -> bool:
        """Create a new role"""
        try:
            if role.name in self.roles:
                return False
            
            self.roles[role.name] = role
            
            # Store in Redis
            role_data = {
                "name": role.name,
                "description": role.description,
                "permissions": [str(perm) for perm in role.permissions],
                "parent_roles": list(role.parent_roles),
                "is_system_role": role.is_system_role,
                "created_at": role.created_at.isoformat()
            }
            
            await self.redis.hset(
                f"rbac:role:{role.name}",
                mapping={k: json.dumps(list(v)) if isinstance(v, set) else 
                           json.dumps(v) if isinstance(v, (list, bool)) else 
                           v.isoformat() if isinstance(v, datetime) else 
                           str(v) 
                         for k, v in role_data.items()}
            )
            
            logger.info(f"Created role: {role.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create role {role.name}: {e}")
            return False
    
    async def delete_role(self, role_name: str) -> bool:
        """Delete a role"""
        try:
            if role_name not in self.roles:
                return False
            
            role = self.roles[role_name]
            if role.is_system_role:
                logger.warning(f"Cannot delete system role: {role_name}")
                return False
            
            # Remove from memory
            del self.roles[role_name]
            
            # Remove from Redis
            await self.redis.delete(f"rbac:role:{role_name}")
            
            # Remove role from all users
            for user_id in self.user_roles:
                self.user_roles[user_id].discard(role_name)
                await self._update_user_roles_in_redis(user_id)
            
            logger.info(f"Deleted role: {role_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete role {role_name}: {e}")
            return False
    
    async def assign_role_to_user(self, user_id: str, role_name: str) -> bool:
        """Assign role to user"""
        try:
            if role_name not in self.roles:
                return False
            
            if user_id not in self.user_roles:
                self.user_roles[user_id] = set()
            
            self.user_roles[user_id].add(role_name)
            await self._update_user_roles_in_redis(user_id)
            
            logger.info(f"Assigned role {role_name} to user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to assign role {role_name} to user {user_id}: {e}")
            return False
    
    async def revoke_role_from_user(self, user_id: str, role_name: str) -> bool:
        """Revoke role from user"""
        try:
            if user_id not in self.user_roles:
                return False
            
            self.user_roles[user_id].discard(role_name)
            await self._update_user_roles_in_redis(user_id)
            
            logger.info(f"Revoked role {role_name} from user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to revoke role {role_name} from user {user_id}: {e}")
            return False
    
    async def check_permission(
        self,
        user_id: str,
        resource_type: ResourceType,
        action: Action,
        resource_id: str = None,
        context: Dict[str, Any] = None
    ) -> bool:
        """Check if user has permission for specific action"""
        try:
            user_roles = self.user_roles.get(user_id, set())
            if not user_roles:
                return False
            
            # Get all permissions from user's roles (including inherited)
            all_permissions = set()
            for role_name in user_roles:
                role_permissions = await self._get_role_permissions_recursive(role_name)
                all_permissions.update(role_permissions)
            
            # Check if any permission matches
            for permission in all_permissions:
                if permission.matches(resource_type, action, resource_id):
                    # Check additional conditions if present
                    if await self._evaluate_permission_conditions(permission, context):
                        return True
            
            return False
            
        except Exception as e:
            logger.error(f"Permission check failed for user {user_id}: {e}")
            return False
    
    async def _get_role_permissions_recursive(self, role_name: str) -> Set[Permission]:
        """Get all permissions for role including inherited permissions"""
        if role_name not in self.roles:
            return set()
        
        role = self.roles[role_name]
        permissions = set(role.permissions)
        
        # Add permissions from parent roles
        for parent_role_name in role.parent_roles:
            parent_permissions = await self._get_role_permissions_recursive(parent_role_name)
            permissions.update(parent_permissions)
        
        return permissions
    
    async def _evaluate_permission_conditions(
        self,
        permission: Permission,
        context: Dict[str, Any] = None
    ) -> bool:
        """Evaluate additional conditions for permission"""
        if not permission.conditions or not context:
            return True
        
        try:
            # Time-based conditions
            if "time_range" in permission.conditions:
                time_range = permission.conditions["time_range"]
                current_time = datetime.utcnow().time()
                start_time = datetime.strptime(time_range["start"], "%H:%M").time()
                end_time = datetime.strptime(time_range["end"], "%H:%M").time()
                
                if not (start_time <= current_time <= end_time):
                    return False
            
            # IP-based conditions
            if "allowed_ips" in permission.conditions:
                user_ip = context.get("ip_address")
                if user_ip not in permission.conditions["allowed_ips"]:
                    return False
            
            # Resource owner conditions
            if "owner_only" in permission.conditions and permission.conditions["owner_only"]:
                resource_owner = context.get("resource_owner")
                user_id = context.get("user_id")
                if resource_owner != user_id:
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Condition evaluation failed: {e}")
            return False
    
    async def _update_user_roles_in_redis(self, user_id: str):
        """Update user roles in Redis"""
        user_roles = list(self.user_roles.get(user_id, set()))
        await self.redis.hset(
            f"rbac:user:{user_id}",
            "roles",
            json.dumps(user_roles)
        )
    
    async def get_user_permissions(self, user_id: str) -> List[str]:
        """Get all permissions for user"""
        user_roles = self.user_roles.get(user_id, set())
        all_permissions = set()
        
        for role_name in user_roles:
            role_permissions = await self._get_role_permissions_recursive(role_name)
            all_permissions.update(role_permissions)
        
        return [str(perm) for perm in all_permissions]
    
    async def get_user_roles(self, user_id: str) -> List[str]:
        """Get roles assigned to user"""
        return list(self.user_roles.get(user_id, set()))
    
    async def load_from_redis(self):
        """Load RBAC data from Redis"""
        try:
            # Load roles
            role_keys = await self.redis.keys("rbac:role:*")
            for key in role_keys:
                role_data = await self.redis.hgetall(key)
                if role_data:
                    role_name = role_data["name"]
                    
                    # Skip if system role already exists
                    if role_name in self.roles and self.roles[role_name].is_system_role:
                        continue
                    
                    # Parse permissions
                    permissions = set()
                    perm_strings = json.loads(role_data.get("permissions", "[]"))
                    for perm_str in perm_strings:
                        # Parse permission string (resource_type:action or resource_type:resource_id:action)
                        parts = perm_str.split(":")
                        if len(parts) >= 2:
                            resource_type = ResourceType(parts[0])
                            action = Action(parts[-1])
                            resource_id = parts[1] if len(parts) == 3 else None
                            permissions.add(Permission(resource_type, action, resource_id))
                    
                    role = Role(
                        name=role_name,
                        description=role_data.get("description", ""),
                        permissions=permissions,
                        parent_roles=set(json.loads(role_data.get("parent_roles", "[]"))),
                        is_system_role=json.loads(role_data.get("is_system_role", "false")),
                        created_at=datetime.fromisoformat(role_data.get("created_at", datetime.utcnow().isoformat()))
                    )
                    
                    self.roles[role_name] = role
            
            # Load user roles
            user_keys = await self.redis.keys("rbac:user:*")
            for key in user_keys:
                user_id = key.decode().split(":")[-1]
                user_data = await self.redis.hgetall(key)
                if user_data and "roles" in user_data:
                    roles = set(json.loads(user_data["roles"]))
                    self.user_roles[user_id] = roles
            
            logger.info("Loaded RBAC data from Redis")
            
        except Exception as e:
            logger.error(f"Failed to load RBAC data from Redis: {e}")
    
    async def create_custom_permission(
        self,
        resource_type: ResourceType,
        action: Action,
        resource_id: str = None,
        conditions: Dict[str, Any] = None
    ) -> Permission:
        """Create custom permission with conditions"""
        return Permission(
            resource_type=resource_type,
            action=action,
            resource_id=resource_id,
            conditions=conditions or {}
        )