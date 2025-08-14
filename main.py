#!/usr/bin/env python3
"""
HDB Polynomial Price Calculator - Main Entry Point
Enhanced HDB valuation with 4th degree polynomial regression
"""

import sys
import os
from cli_interface import SimplifiedHDBCalculatorCLI

def main():
    """Main entry point for the HDB Polynomial Price Calculator"""
    try:
        print("🔄 Starting HDB Polynomial Price Calculator...")
        
        # Initialize the simplified CLI interface
        calculator = SimplifiedHDBCalculatorCLI()
        
        # Validate system requirements
        try:
            import pandas, sklearn, matplotlib, seaborn, numpy, rich
            print("✅ All required packages available")
        except ImportError as e:
            print(f"❌ Missing required package: {e}")
            print("💡 Please install required dependencies and restart")
            sys.exit(1)
        
        # Check if data files exist
        if not os.path.exists('sample_data.csv'):
            print("⚠️ Main data file not found - please ensure sample_data.csv is available")
            print("💡 The application will attempt to load the training data")
        
        # Create output directories
        os.makedirs('graphs', exist_ok=True)
        os.makedirs('exports', exist_ok=True)
        
        # Start the application
        calculator.run()
        
    except KeyboardInterrupt:
        print("\n\n👋 Thank you for using HDB Polynomial Price Calculator!")
        print("💡 Application terminated by user")
        sys.exit(0)
    except FileNotFoundError as e:
        print(f"\n❌ File not found error: {e}")
        print("💡 Please ensure all required files are present")
        sys.exit(1)
    except ImportError as e:
        print(f"\n❌ Import error: {e}")
        print("💡 Please check that all required packages are installed")
        sys.exit(1)
    except MemoryError:
        print("\n❌ Memory error: Insufficient system memory")
        print("💡 Close other applications and try again")
        sys.exit(1)
    except PermissionError as e:
        print(f"\n❌ Permission error: {e}")
        print("💡 Check file permissions and run with appropriate access")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")
        print("💡 Please restart the application and try again")
        print("📝 If the problem persists, check the error logs")
        sys.exit(1)

if __name__ == "__main__":
    main()
