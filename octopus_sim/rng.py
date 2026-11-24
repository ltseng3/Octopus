"""Policy-order-independent random draws based on semantic keys."""
import hashlib
import json


def uniform(seed: int, *keys: object) -> float:
    data = json.dumps([seed, *keys], separators=(',', ':'), ensure_ascii=True).encode()
    digest = hashlib.blake2b(data, digest_size=8, person=b'octosim1').digest()
    # 53 random bits, exactly representable as a float fraction.
    return (int.from_bytes(digest, 'big') >> 11) / (1 << 53)
