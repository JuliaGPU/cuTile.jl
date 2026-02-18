# Token role enum
@enum TokenRole LAST_OP LAST_STORE

# Acquire token key (singleton)
struct AcquireTokenKey end
const ACQUIRE_TOKEN_KEY = AcquireTokenKey()

# Alias token key (per alias set and role)
struct AliasTokenKey
    alias_set::AliasSet
    role::TokenRole
end

# Union type for all token keys
const TokenKey = Union{AliasTokenKey, AcquireTokenKey}

# Helper constructors
"""
    last_op_key(alias_set::AliasSet) -> AliasTokenKey

Create a TokenKey for the last operation (load or store) on an alias set.
"""
last_op_key(alias_set::AliasSet) = AliasTokenKey(alias_set, LAST_OP)

"""
    last_store_key(alias_set::AliasSet) -> AliasTokenKey

Create a TokenKey for the last store operation on an alias set.
"""
last_store_key(alias_set::AliasSet) = AliasTokenKey(alias_set, LAST_STORE)

# Make TokenKey hashable for use in Dict
Base.hash(key::AliasTokenKey, h::UInt) = hash((key.alias_set, key.role), h)
Base.:(==)(a::AliasTokenKey, b::AliasTokenKey) =
    a.alias_set == b.alias_set && a.role == b.role

Base.hash(::AcquireTokenKey, h::UInt) = hash(:ACQUIRE_TOKEN_KEY, h)
Base.:(==)(::AcquireTokenKey, ::AcquireTokenKey) = true
