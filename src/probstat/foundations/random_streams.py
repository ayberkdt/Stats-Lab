"""Deterministic, splittable seed derivation for reproducible random streams.

Every stochastic routine in the library accepts a call-local seed. Large
studies need many independent streams (inputs, MCMC chains, bootstrap
replicates, filters) without hand-picking seeds such as ``310`` and ``311``.
:class:`RandomStream` derives child seeds from a root seed and a path of keys
by SplitMix64 mixing, so streams are reproducible, order-independent, and do
not depend on Python's salted string hashing.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

_MASK64 = (1 << 64) - 1
_GOLDEN_GAMMA = 0x9E3779B97F4A7C15


def splitmix64(state: int) -> int:
    """Apply the SplitMix64 output finalizer to one 64-bit state.

    Args:
        state: Integer state; it is reduced modulo ``2**64``.

    Returns:
        Well-mixed 64-bit unsigned integer.

    Raises:
        TypeError: If ``state`` is not an integer.

    Equation:
        [EQ-RNG-SPLITMIX-001].

    References:
        [REF-STEELE-LEA-FLOOD-2014].

    """
    if isinstance(state, bool) or not isinstance(state, int):
        raise TypeError("state must be an integer")
    z = (state + _GOLDEN_GAMMA) & _MASK64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK64
    return z ^ (z >> 31)


def _key_to_int(key: int | str) -> int:
    """Encode an integer or text key as a deterministic 64-bit integer.

    Args:
        key: Non-negative integer or text label.

    Returns:
        64-bit integer independent of interpreter hash salting.

    Raises:
        TypeError: If ``key`` is neither an integer nor a string.
        ValueError: If an integer key is negative.

    References:
        [EQ-RNG-SPLITMIX-001], [REF-STEELE-LEA-FLOOD-2014].

    """
    if isinstance(key, bool):
        raise TypeError("stream keys must be integers or strings")
    if isinstance(key, int):
        if key < 0:
            raise ValueError("integer stream keys must be non-negative")
        return splitmix64(key & _MASK64)
    if isinstance(key, str):
        state = 0x6A09E667F3BCC909
        for byte in key.encode("utf-8"):
            state = splitmix64(state ^ byte)
        return state
    raise TypeError("stream keys must be integers or strings")


def derive_seed(root: int, *path: int | str) -> int:
    """Derive a reproducible child seed from a root seed and a key path.

    Args:
        root: Non-negative root seed.
        *path: Integer or text keys naming the child stream.

    Returns:
        Non-negative 64-bit seed; distinct paths give unrelated seeds.

    Raises:
        TypeError: If the root or a key has an invalid type.
        ValueError: If the root or an integer key is negative.

    Equation:
        [EQ-RNG-SPLITMIX-001].

    References:
        [REF-STEELE-LEA-FLOOD-2014].

    """
    if isinstance(root, bool) or not isinstance(root, int):
        raise TypeError("root seed must be an integer")
    if root < 0:
        raise ValueError("root seed must be non-negative")
    state = splitmix64(root & _MASK64)
    for key in path:
        state = splitmix64(state ^ _key_to_int(key))
    return state


@dataclass(frozen=True, slots=True)
class RandomStream:
    """A named node in a tree of reproducible pseudo-random streams.

    Args:
        seed: Non-negative root seed for this node.

    References:
        [REF-STEELE-LEA-FLOOD-2014].

    """

    seed: int

    def __post_init__(self) -> None:
        """Validate the seed.

        Raises:
            TypeError: If the seed is not an integer.
            ValueError: If the seed is negative.

        References:
            Project validation policy. No equation is evaluated.

        """
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")

    def child(self, *path: int | str) -> RandomStream:
        """Return the child stream addressed by a key path.

        Args:
            *path: Integer or text keys such as ``"inputs"`` or a chain index.

        Returns:
            Independent-by-construction child stream.

        Equation:
            [EQ-RNG-SPLITMIX-001].

        References:
            [REF-STEELE-LEA-FLOOD-2014].

        """
        return RandomStream(derive_seed(self.seed, *path))

    def spawn(self, count: int, *, label: str = "spawn") -> tuple[RandomStream, ...]:
        """Return ``count`` numbered child streams under one label.

        Args:
            count: Non-negative number of children.
            label: Text key shared by the children.

        Returns:
            Tuple of child streams ``child(label, 0)``, ``child(label, 1)``, ...

        Raises:
            ValueError: If ``count`` is negative.

        Equation:
            [EQ-RNG-SPLITMIX-001].

        References:
            [REF-STEELE-LEA-FLOOD-2014].

        """
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("count must be a non-negative integer")
        return tuple(self.child(label, index) for index in range(count))

    def generator(self) -> random.Random:
        """Return a fresh Mersenne Twister generator seeded by this node.

        Returns:
            A new :class:`random.Random`; repeated calls restart the stream.

        References:
            [REF-MATSUMOTO-NISHIMURA-1998].

        """
        return random.Random(self.seed)

    def integer_seed(self) -> int:
        """Return a seed suitable for APIs that accept ``seed: int``.

        Returns:
            This node's 64-bit seed.

        References:
            [REF-STEELE-LEA-FLOOD-2014].

        """
        return self.seed


def resolve_stream(seed: int | RandomStream | None) -> RandomStream:
    """Normalize ``None``, an integer seed, or a stream into a stream.

    ``None`` draws fresh operating-system entropy, which is explicitly not
    reproducible; pass an integer or stream for reproducible work.

    Args:
        seed: Optional seed or existing stream.

    Returns:
        A :class:`RandomStream`.

    Raises:
        TypeError: If ``seed`` has an unsupported type.

    References:
        [REF-STEELE-LEA-FLOOD-2014].

    """
    if seed is None:
        return RandomStream(random.SystemRandom().getrandbits(63))
    if isinstance(seed, RandomStream):
        return seed
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer, RandomStream, or None")
    return RandomStream(seed)
