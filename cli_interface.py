# SOURITRA SAMANTA (3C)

# Import local modules from other libraries n stuff
import os  
from typing import Dict, Optional 
from rich.console import Console  
from rich.table import Table
from rich.panel import Panel  
from rich import print as rprint 
from rich.prompt import Prompt, IntPrompt, FloatPrompt  
import pandas as pd  

# Import the custom .py modules that we created
from hdb_polynomial_model import HDBPolynomialPriceModel  
from data_processor import HDBDataProcessor
from visualizer import HDBVisualizer  

class SimplifiedHDBCalculatorCLI:
    def __init__(self):
        self.console = Console()  # Rich console for printing
        self.model = HDBPolynomialPriceModel()  # Instantiate HDB model
        self.processor = HDBDataProcessor()  # Instantiate data processor
        self.visualizer = HDBVisualizer()  # Instantiate visualizer
        self.session_predictions = []  # Store session predictions

    # Makes console clear when needed
    def clear_screen(self):
        os.system('cls' if os.name == 'nt' else 'clear')  

    # Show the cool banner thingy (it's called ascii art)
    def display_banner(self):
        """Display application banner"""
        banner = """
 _   _ ____  ____   __     __    _             _   _             
| | | |  _ \| __ )  \ \   / /_ _| |_   _  __ _| |_(_) ___  _ __  
| |_| | | | |  _ \   \ \ / / _` | | | | |/ _` | __| |/ _ \| '_ \ 
|  _  | |_| | |_) |   \ V / (_| | | |_| | (_| | |_| | (_) | | | |
|_| |_|____/|____/     \_/ \__,_|_|\__,_|\__,_|\__|_|\___/|_| |_|
 / ___|__ _| | ___ _   _| | __ _| |_ ___  _ __                   
| |   / _` | |/ __| | | | |/ _` | __/ _ \| '__|                  
| |__| (_| | | (__| |_| | | (_| | || (_) | |                     
 \____\__,_|_|\___|\__,_|_|\__,_|\__\___/|_|                     
        """
        self.console.print(banner, style="bold cyan")
        
    # Main menu panel
    def show_main_menu(self):
        menu_panel = Panel(
            """[bold green]1.[/bold green] Calculate HDB Price
[bold green]2.[/bold green] View Results History  
[bold green]3.[/bold green] Exit""",
            title="[bold blue]Main Menu[/bold blue]",
            border_style="blue"
        ) 
        self.console.print(menu_panel)
        
    # Loads data, cleans it, trains the polynomial model, and shows metrics (🥶)
    def load_and_train_model(self):
        with self.console.status("[bold green]Loading data and training polynomial model..."):
            self.model.load_data('sample_data.csv')  # Load CSV data into model
            if self.model.df is not None:
                cleaned_data = self.processor.clean_data(self.model.df)  # Clean dataset
                self.model.df = cleaned_data  # Replace original df with cleaned df
            self.model.train_model()  # Train polynomial model
            metrics = self.model.get_model_metrics()  # Get model metrics
            polynomial_info = self.model.get_polynomial_equation_info()  # Get polynomial info
            summary_table = Table(title="Model Training Summary")  # Create summary table
            summary_table.add_column("Metric", style="cyan")
            summary_table.add_column("Value", style="green")
            summary_table.add_row("Polynomial Degree", str(polynomial_info['degree']))
            summary_table.add_row("Training Samples", f"{metrics['n_samples']:,}")
            summary_table.add_row("Original Features", str(metrics['n_features']))
            summary_table.add_row("Polynomial Features", f"{polynomial_info['n_features']:,}")
            summary_table.add_row("Training R²", f"{metrics['train_r2']:.4f}")
            summary_table.add_row("Testing R²", f"{metrics['test_r2']:.4f}")
            summary_table.add_row("Test MAE", f"SGD ${metrics['test_mae']:,.2f}")
            summary_table.add_row("Test RMSE", f"SGD ${metrics['test_rmse']:,.2f}")
            self.console.print(summary_table)  # Display metrics
            rprint("\n[bold green]✅ Model trained successfully![/bold green]")  

    # Price predict tool (this is basically the main thing 🔥)
    def predict_price(self):
        if not self.model.is_trained:
            rprint("[bold red]❌ Model not trained. Please train the model first.[/bold red]")
            return  # Exit if model is not trained
        rprint("[bold blue]🏠 Calculate HDB Price[/bold blue]")  # Section header
        inputs = self.collect_user_inputs()  # Collect user inputs
        if inputs:
            is_valid, errors = self.processor.validate_input_data(inputs)  # Validate inputs
            if not is_valid:
                for error in errors:
                    print(f"❌ {error}") 
                return
            prediction, contributions = self.model.predict_price(inputs)  # Make prediction
            self.display_prediction_results(inputs, prediction, contributions)
            
            # Generate visualizations automatically
            rprint("\n[bold cyan]📊 Generating visualization charts...[/bold cyan]")
            with self.console.status("[bold green]Creating charts, graphs, and heatmaps..."):
                generated_files = self.visualizer.generate_prediction_summary_visuals(
                    self.model, inputs, prediction, contributions
                )
            
            # Display generated files info
            files_panel = Panel(
                f"[bold green]✅ Generated Files:[/bold green]\n" +
                "\n".join([f"📁 {file}" for file in generated_files]),
                title="[bold blue]Visualization Files Created[/bold blue]",
                border_style="green"
            )
            self.console.print(files_panel)
            
            self.session_predictions.append({
                'inputs': inputs.copy(),
                'prediction': prediction,
                'timestamp': pd.Timestamp.now()
            })  # Save prediction in session
            
    # Gathers and validates property details from the user (this part is kinda confusing ngl)
    def collect_user_inputs(self) -> Optional[Dict]:
        inputs = {}  # Initialize input dictionary
        towns = self.model.get_available_towns()  # Get list of towns
        if not towns:
            print("❌ No towns available")
            return None
            # souritra wrote this (watermark)
        rprint(f"[bold cyan]Towns ({len(towns)} available):[/bold cyan]")
        for i, town in enumerate(towns, 1):
            rprint(f"  {i}. {town}")  # Display towns with index
        rprint(f"\n[bold yellow]💡 You can type the town number (1-{len(towns)}) or town name[/bold yellow]")
        town_input = Prompt.ask("Select town")  # Ask for town input
        try:
            town_choice = int(town_input)
            if 1 <= town_choice <= len(towns):
                inputs['town'] = towns[town_choice - 1]  # Choose town by number
            else:
                print(f"❌ Please enter a number between 1-{len(towns)}")
                return None
        except ValueError:
            matching_towns = [t for t in towns if town_input.upper() in t.upper()]  # Match name
            if not matching_towns:
                print("❌ Town not found")
                return None
            elif len(matching_towns) == 1:
                inputs['town'] = matching_towns[0]  # Single match
            else:
                rprint(f"[bold yellow]Multiple matches found:[/bold yellow]")
                for i, town in enumerate(matching_towns[:5], 1):
                    rprint(f"  {i}. {town}")  # Show top matches
                choice = IntPrompt.ask("Select number", choices=[str(i) for i in range(1, min(6, len(matching_towns) + 1))])
                inputs['town'] = matching_towns[choice - 1]  # Choose from matches
        flat_types = self.model.get_available_flat_types()  # Get flat types
        rprint(f"[bold cyan]Flat Types:[/bold cyan]")
        for i, ft in enumerate(flat_types, 1):
            rprint(f"  {i}. {ft}")  # Show flat types
        ft_choice = IntPrompt.ask("Select flat type", choices=[str(i) for i in range(1, len(flat_types) + 1)])
        inputs['flat_type'] = flat_types[ft_choice - 1]  # Save flat type
        rprint("[bold cyan]💡 Typical ranges: 2-room (34-64 sqm), 3-room (60-90 sqm), 4-room (80-120 sqm)[/bold cyan]")
        while True:
            area = FloatPrompt.ask("Enter floor area (sqm)")  # Ask floor area
            if 30 <= area <= 300:
                inputs['floor_area_sqm'] = area
                break
            else:
                print("❌ Floor area must be between 30-300 sqm")
        storey_ranges = self.model.get_available_storey_ranges()  # Get storey ranges
        rprint(f"[bold cyan]Storey Ranges:[/bold cyan]")
        for i, sr in enumerate(storey_ranges, 1):
            rprint(f"  {i}. {sr}")  # Display storey ranges
        sr_choice = IntPrompt.ask("Select storey range", choices=[str(i) for i in range(1, len(storey_ranges) + 1)])
        inputs['storey_range'] = storey_ranges[sr_choice - 1]  # Save storey range
        common_models = ['Model A', 'Improved', 'New Generation', 'Premium Apartment', 'Standard', 'Apartment']
        rprint(f"[bold cyan]Flat Models:[/bold cyan]")
        for i, model in enumerate(common_models, 1):
            rprint(f"  {i}. {model}")  # Show common flat models
        model_choice = IntPrompt.ask("Select flat model", choices=[str(i) for i in range(1, len(common_models) + 1)])
        inputs['flat_model'] = common_models[model_choice - 1]  # Save flat model
        rprint("[bold cyan]💡 Most HDB flats have 50-90 years remaining lease[/bold cyan]")
        while True:
            lease = IntPrompt.ask("Enter remaining lease (years)")  # Ask remaining lease
            if 45 <= lease <= 99:
                inputs['remaining_lease'] = lease
                break
            else:
                print("❌ Remaining lease must be between 45-99 years for accurate predictions")
        return inputs  # Return collected inputs
            
    # Displays predicted price, inputs, and top feature contributions (summary)        
    def display_prediction_results(self, inputs: Dict, prediction: float, contributions: Dict):
        self.clear_screen()  # Clear console
        self.display_banner()  # Show banner
        result_panel = Panel(f"[bold green]SGD ${prediction:,.2f}[/bold green]", title="[bold blue]Predicted HDB Price[/bold blue]", border_style="green")
        self.console.print(result_panel)  # Show predicted price
        input_table = Table(title="Input Summary")  # Create table for inputs
        input_table.add_column("Property", style="cyan")
        input_table.add_column("Value", style="white")
        for key, value in inputs.items():
            display_key = key.replace('_', ' ').title()
            input_table.add_row(display_key, str(value))  # Fill table with input values
        self.console.print(input_table)  # Print input table

    # View session predictions
    def view_history(self):
        if not self.session_predictions:
            print("⚠️ No previous predictions available")
            return
        history_table = Table(title="Session Predictions History")
        history_table.add_column("Timestamp", style="cyan")
        history_table.add_column("Town", style="green")
        history_table.add_column("Flat Type", style="magenta")
        history_table.add_column("Floor Area (sqm)", style="yellow")
        history_table.add_column("Predicted Price (SGD)", style="green")
        for record in self.session_predictions:
            history_table.add_row(
                str(record['timestamp']),
                record['inputs'].get('town', ''),
                record['inputs'].get('flat_type', ''),
                str(record['inputs'].get('floor_area_sqm', '')),
                f"${record['prediction']:,.2f}"
            )
        self.console.print(history_table)

    # Run CLI itself
    def run(self):
        self.clear_screen()
        self.display_banner()
        self.load_and_train_model()
        
        # Wait 5 seconds then clear screen for cleaner interface
        import time
        rprint("\n[bold cyan]🎉 Ready to use! Starting interface in 5 seconds...[/bold cyan]")
        time.sleep(5)
        self.clear_screen()
        self.display_banner()
        
        while True:
            self.show_main_menu()
            choice = Prompt.ask("[bold blue]Select an option[/bold blue]", choices=["1", "2", "3"])
            if choice == "1":
                self.predict_price()
            elif choice == "2":
                self.view_history()
            elif choice == "3":
                rprint("[bold green]✅ Exiting HDB Calculator. Bye![/bold green]")
                break
            else:
                print("❌ Invalid choice")

# SOURITRA SAMANTA (3C)
