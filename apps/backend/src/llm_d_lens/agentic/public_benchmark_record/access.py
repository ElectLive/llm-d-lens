"""Access to the public corpus requires an authenticated, global query grant.

Public describes the evidence sources, not caller visibility. Cluster ownership
and resource shares do not apply, and disabled-auth identities are not callers.
"""

from __future__ import annotations

from dataclasses import replace

from llm_d_lens.agentic.public_benchmark_record.query_contracts import QUERY_PERMISSION
from llm_d_lens.auth.contracts import Principal, ScopeType
from llm_d_lens.auth.policy import authorize
from llm_d_lens.core.exceptions import ForbiddenError, PasswordChangeRequiredError, UnauthenticatedError


def require_query_access(principal: Principal | None) -> None:
    """Require a real caller and GLOBAL query permission before corpus access.

    Global admins retain the common policy bypass. Other roles, including
    maintainer and end-user, require an explicit global grant.
    """
    _require_global_access(principal, QUERY_PERMISSION)


def _require_global_access(principal: Principal | None, permission: str) -> None:
    if principal is None or not principal.user_id or (principal.user_id == "system" and principal.authz_version == 0):
        raise UnauthenticatedError("authentication required")
    if principal.must_change_password:
        raise PasswordChangeRequiredError("password change required before continuing")
    bindings = tuple(binding for binding in principal.bindings if binding.scope_type is ScopeType.GLOBAL)
    roles = {binding.role for binding in bindings}
    global_principal = replace(
        principal,
        bindings=bindings,
        # Casbin subjects and roles share a namespace; remove unbound policies too.
        role_permissions={role: codes for role, codes in principal.role_permissions.items() if role in roles},
    )
    if not authorize(global_principal, permission).allowed:
        raise ForbiddenError("permission denied")
