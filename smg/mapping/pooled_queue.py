from __future__ import annotations

import collections
import pytypes
import threading

from typing import Callable, Deque, Generic, List, Optional, TypeVar


# TYPE VARIABLE

T = TypeVar('T')


# MAIN CLASS

class PooledQueue(Generic[T]):
    """A queue that is backed by a pool of reusable elements."""

    # NESTED TYPES

    class EPoolEmptyStrategy(int):
        """Used to specify what should happen when a push is attempted on a pooled queue with an empty pool."""
        pass

    # Discard the new element.
    PES_DISCARD: EPoolEmptyStrategy = 0
    # Add an extra element to the pool to accommodate the new element.
    PES_GROW: EPoolEmptyStrategy = 1
    # Move a random element from the queue back to the pool to accommodate the new element.
    PES_REPLACE_RANDOM: EPoolEmptyStrategy = 2
    # Wait for another thread to pop an element from the queue, thereby making space for the new element.
    PES_WAIT: EPoolEmptyStrategy = 3

    class PushHandler:
        """Used to handle the process of pushing an element onto the queue."""

        # CONSTRUCTOR

        def __init__(self, base: PooledQueue[T], elt: Optional[T]):
            """
            Construct a push handler.

            :param base:    The pooled queue on which the push was called.
            :param elt:     The element that is to be pushed onto the queue (if any).
            """
            self.__base: PooledQueue[T] = base
            self.__elt: Optional[T] = elt

        # SPECIAL METHODS

        def __enter__(self):
            """Dummy __enter__ method to allow the push handler to be used in a with statement."""
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            """Complete the push by pushing the element (if any) onto the queue."""
            if self.__elt is not None:
                self.__base._end_push(self.__elt)

        # PUBLIC METHODS

        def get(self) -> Optional[T]:
            """
            Get the element that is to be pushed onto the queue (if any).

            :return:    The element that is to be pushed onto the queue (if any).
            """
            return self.__elt

    # CONSTRUCTOR

    def __init__(self, pool_empty_strategy: EPoolEmptyStrategy = PES_GROW):
        """
        Construct a pooled queue.

        :param pool_empty_strategy: What should happen when a push is attempted while the pool is empty.
        """
        self.__pool_empty_strategy: PooledQueue.EPoolEmptyStrategy = pool_empty_strategy
        if pool_empty_strategy == PooledQueue.PES_REPLACE_RANDOM:
            # TODO
            pass

        self.__lock: threading.Lock = threading.Lock()
        self.__maker: Optional[Callable[[], T]] = None
        self.__pool: Deque[T] = collections.deque()
        self.__pool_non_empty: threading.Condition = threading.Condition(self.__lock)
        self.__queue: Deque[T] = collections.deque()
        self.__queue_non_empty: threading.Condition = threading.Condition(self.__lock)

    # PUBLIC METHODS

    def begin_push(self) -> PushHandler:
        """
        Start a push operation.

        .. note::
            In a pooled queue, a push operation is not instantaneous. First, the caller calls begin_push().
            This returns a push handler, which allows the caller to access the element (if any) that is to
            be pushed onto the queue. The caller then writes into this element, rather than constructing a
            new element from scratch. Finally, the __exit__ function of the push handler calls end_push()
            to actually push the element onto the queue.

        :return:    A push handler that will handle the process of pushing an element onto the queue.
        """
        with self.__lock:
            # The first task is to make sure that the pool contains an element into which the caller can write.
            # If the pool is currently empty, we have various options: (i) prevent the push by returning a null
            # element into which to write; (ii) create a new element and add it to the pool; (iii) move a random
            # element from the queue back to the pool (thereby allowing the new element to replace it); or (iv)
            # block until another thread pops an element from the queue and re-adds it to the pool. We choose
            # between these options by specifying a pool empty strategy when the pooled queue is constructed.
            if len(self.__pool) == 0:
                if self.__pool_empty_strategy == PooledQueue.PES_DISCARD:
                    return PooledQueue.PushHandler(self, None)
                elif self.__pool_empty_strategy == PooledQueue.PES_GROW:
                    self.__pool.append(self.__maker())
                elif self.__pool_empty_strategy == PooledQueue.PES_REPLACE_RANDOM:
                    # TODO
                    pass
                elif self.__pool_empty_strategy == PooledQueue.PES_WAIT:
                    while len(self.__pool) == 0:
                        self.__pool_non_empty.wait(0.1)

            # At this point, the pool definitely contains at least one element, so we can simply
            # remove the first element in the pool and return it to the caller for writing.
            elt: T = self.__pool.popleft()
            return PooledQueue.PushHandler(self, elt)

    def empty(self) -> bool:
        """
        Get whether or not the queue is empty.

        :return:    True, if the queue is empty, or False otherwise.
        """
        with self.__lock:
            return len(self.__queue) == 0

    def initialise(self, capacity: int, maker: Optional[Callable[[], T]] = None) -> None:
        """
        Initialise the pool backing the queue.

        .. note::
            If no maker is specified, the default constructor for T is used.

        :param capacity:    The initial capacity of the pool (if we're using the 'grow' strategy, this may change).
        :param maker:       A function that can be used to construct new elements.
        """
        with self.__lock:
            if maker is None:
                # TODO: This should be moved somewhere more central.
                maker = pytypes.type_util.get_orig_class(self).__args__[0]

            self.__maker = maker
            for i in range(capacity):
                self.__pool.append(maker())

    def peek(self) -> T:
        """
        Get the first element in the queue.

        .. note::
            This will block until the queue is non-empty.

        :return:    The first element in the queue.
        """
        with self.__lock:
            while len(self.__queue) == 0:
                self.__queue_non_empty.wait(0.1)
            return self.__queue[0]

    def pop(self) -> None:
        """
        Pop the first element from the queue and return it to the pool.

        .. note::
            This will block until the queue is non-empty.
        """
        with self.__lock:
            while len(self.__queue) == 0:
                self.__queue_non_empty.wait(0.1)
            self.__pool.append(self.__queue.popleft())
            self.__pool_non_empty.notify()

    def size(self) -> int:
        """
        Get the size of the queue.

        :return:    The size of the queue.
        """
        with self.__lock:
            return len(self.__queue)

    # PROTECTED METHODS

    def _end_push(self, elt: T) -> None:
        """
        Complete a push operation by pushing the specified element onto the queue.

        .. note::
            This is called automatically when the push handler associated with the push is destroyed.

        :param elt: The element to be pushed onto the queue.
        """
        with self.__lock:
            self.__queue.append(elt)
            self.__queue_non_empty.notify()
