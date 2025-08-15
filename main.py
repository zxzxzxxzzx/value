# SOURITRA SAMANTA (3C)

import sys
import os
from cli_interface import SimplifiedHDBCalculatorCLI

def main():
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
        
        # Start the application
        calculator.run()
        
    except KeyboardInterrupt:
        print("\n\n👋 Thank you for using HDB Polynomial Price Calculator!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("💡 Please restart the application and try again")
        sys.exit(1)

if __name__ == "__main__":
    main()

# SOURITRA SAMANTA (3C)
