# Directory Structure
ai_environmental_impact/
├── src/
│   ├── python/
│   │   ├── carbon_calculator.py
│   │   ├── data_processor.py
│   │   └── __init__.py
│   ├── rust/
│   │   └── metrics/
│   │       ├── Cargo.toml
│   │       └── src/
│   │           └── lib.rs
│   └── clarity/
│       └── contracts/
│           └── carbon-tracker.clar
├── tests/
│   ├── python/
│   │   └── test_carbon_calculator.py
│   └── rust/
│       └── test_metrics.rs
├── README.md
└── requirements.txt

# carbon_calculator.py
class CarbonFootprintCalculator:
    def __init__(self):
        self.energy_per_gpu_hour = 1.5  # kWh per GPU hour
        self.carbon_per_kwh = 0.475  # kg CO2 per kWh (global average)

    def calculate_training_emissions(self, gpu_count: int, training_hours: float) -> dict:
        """
        Calculate CO2 emissions for AI model training
        
        Args:
            gpu_count: Number of GPUs used
            training_hours: Duration of training in hours
        
        Returns:
            Dictionary containing emissions data
        """
        energy_consumption = gpu_count * training_hours * self.energy_per_gpu_hour
        carbon_emissions = energy_consumption * self.carbon_per_kwh
        
        return {
            "energy_consumption_kwh": energy_consumption,
            "carbon_emissions_kg": carbon_emissions,
            "gpu_count": gpu_count,
            "training_hours": training_hours
        }

    def calculate_inference_emissions(self, requests_per_hour: int, hours: float) -> dict:
        """
        Calculate CO2 emissions for model inference
        
        Args:
            requests_per_hour: Number of inference requests per hour
            hours: Duration of operation in hours
        
        Returns:
            Dictionary containing emissions data
        """
        energy_per_request = 0.0001  # kWh per inference request
        total_requests = requests_per_hour * hours
        energy_consumption = total_requests * energy_per_request
        carbon_emissions = energy_consumption * self.carbon_per_kwh
        
        return {
            "energy_consumption_kwh": energy_consumption,
            "carbon_emissions_kg": carbon_emissions,
            "total_requests": total_requests,
            "operation_hours": hours
        }

# carbon-tracker.clar
(define-data-var total-emissions uint u0)
(define-data-var emissions-by-model (map principal uint) {})

(define-public (record-emissions (amount uint))
    (let ((current-emissions (var-get total-emissions))
          (sender tx-sender))
        (begin
            (var-set total-emissions (+ current-emissions amount))
            (map-set emissions-by-model
                     sender
                     (+ (default-to u0 (map-get? emissions-by-model sender)) amount))
            (ok true))))

(define-read-only (get-total-emissions)
    (ok (var-get total-emissions)))

(define-read-only (get-model-emissions (model principal))
    (ok (default-to u0 (map-get? emissions-by-model model))))

# lib.rs
use std::sync::atomic::{AtomicU64, Ordering};

pub struct PerformanceMetrics {
    total_operations: AtomicU64,
    energy_consumption: AtomicU64,
}

impl PerformanceMetrics {
    pub fn new() -> Self {
        Self {
            total_operations: AtomicU64::new(0),
            energy_consumption: AtomicU64::new(0),
        }
    }

    pub fn record_operation(&self, operation_energy: u64) {
        self.total_operations.fetch_add(1, Ordering::SeqCst);
        self.energy_consumption.fetch_add(operation_energy, Ordering::SeqCst);
    }

    pub fn get_metrics(&self) -> (u64, u64) {
        (
            self.total_operations.load(Ordering::SeqCst),
            self.energy_consumption.load(Ordering::SeqCst),
        )
    }
}

# test_carbon_calculator.py
import unittest
from src.python.carbon_calculator import CarbonFootprintCalculator

class TestCarbonCalculator(unittest.TestCase):
    def setUp(self):
        self.calculator = CarbonFootprintCalculator()

    def test_training_emissions(self):
        result = self.calculator.calculate_training_emissions(
            gpu_count=4,
            training_hours=24
        )
        self.assertGreater(result["carbon_emissions_kg"], 0)
        self.assertEqual(result["gpu_count"], 4)
        self.assertEqual(result["training_hours"], 24)

    def test_inference_emissions(self):
        result = self.calculator.calculate_inference_emissions(
            requests_per_hour=1000,
            hours=24
        )
        self.assertGreater(result["carbon_emissions_kg"], 0)
        self.assertEqual(result["total_requests"], 24000)