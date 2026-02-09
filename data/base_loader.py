"""
Base Loader Interface

Abstract base class defining the interface for case data loaders.
This allows for consistent handling of single-image and
multi-image (NLMCXR) datasets.
"""

from abc import ABC, abstractmethod
from typing import Union, List, Dict, Any, Tuple
import numpy as np


class BaseNoduleLoader(ABC):
    """
    Abstract base class for nodule/case data loaders.

    This interface allows the system to work with different datasets
    that may have different image structures (single vs multiple images per case).
    """

    @abstractmethod
    def get_case_ids(self) -> List[str]:
        """
        Return list of available case IDs.

        Returns:
            List of case identifier strings
        """
        pass

    @abstractmethod
    def load_case(
        self,
        case_id: str
    ) -> Tuple[Union[np.ndarray, List[np.ndarray]], Dict[str, Any]]:
        """
        Load case data (images and metadata).

        Args:
            case_id: Unique identifier for the case

        Returns:
            Tuple of (images, metadata) where:
            - images: Single np.ndarray for single-image datasets
                    or List[np.ndarray] for multi-image datasets (NLMCXR)
            - metadata: Dict with case features and attributes

        Raises:
            FileNotFoundError: If case_id is not found
        """
        pass

    @property
    @abstractmethod
    def supports_multi_image(self) -> bool:
        """
        Whether this loader returns multiple images per case.

        Returns:
            True if load_case() returns List[np.ndarray], False if single np.ndarray
        """
        pass

    def get_dataset_info(self) -> Dict[str, Any]:
        """
        Get information about the dataset.

        Returns:
            Dict with dataset metadata (optional override)
        """
        return {
            "total_cases": len(self.get_case_ids()),
            "supports_multi_image": self.supports_multi_image
        }


class LoaderFactory:
    """
    Factory for creating appropriate data loaders.

    This allows the system to automatically select the right loader
    based on dataset type.
    """

    _loaders = {}

    @classmethod
    def register_loader(cls, dataset_name: str, loader_class: type):
        """
        Register a loader class for a dataset.

        Args:
            dataset_name: Name of the dataset (e.g., "NLMCXR")
            loader_class: Class implementing BaseNoduleLoader
        """
        cls._loaders[dataset_name.lower()] = loader_class

    @classmethod
    def get_loader(cls, dataset_name: str, **kwargs) -> BaseNoduleLoader:
        """
        Get a loader instance for the specified dataset.

        Args:
            dataset_name: Name of the dataset
            **kwargs: Arguments to pass to loader constructor

        Returns:
            Loader instance

        Raises:
            ValueError: If dataset_name is not registered
        """
        dataset_key = dataset_name.lower()
        if dataset_key not in cls._loaders:
            raise ValueError(
                f"Unknown dataset: {dataset_name}. "
                f"Available: {list(cls._loaders.keys())}"
            )

        loader_class = cls._loaders[dataset_key]
        return loader_class(**kwargs)

    @classmethod
    def list_datasets(cls) -> List[str]:
        """
        List all registered datasets.

        Returns:
            List of dataset names
        """
        return list(cls._loaders.keys())
