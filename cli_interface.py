"""
Simplified CLI Interface for HDB Polynomial Price Prediction
Clean and intuitive console-based interface
"""

import os
import sys
from typing import Dict, List, Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import print as rprint
from rich.prompt import Prompt, IntPrompt, FloatPrompt
import pandas as pd

from hdb_polynomial_model import HDBPolynomialPriceModel
from data_processor import HDBDataProcessor
from visualizer import HDBVisualizer

class SimplifiedHDBCalculatorCLI:
    """Simplified CLI interface for HDB polynomial price prediction"""
    
    def __init__(self):
        self.console = Console()
        self.model = HDBPolynomialPriceModel()
        self.processor = HDBDataProcessor()
        self.visualizer = HDBVisualizer()
        self.session_predictions = []
    
    def clear_screen(self):
        """Clear the console screen"""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def display_banner(self):
        """Display application banner"""
        banner = """
 _   _ ____  ____   __     __    _             _   _             
| | | |  _ \| __ )  \ \   / /_ _| |_   _  __ _| |_(_) ___  _ __  
| |_| | | | |  _ \   \ \ / / _` | | | | |/ _` | __| |/ _ \| '_ \ 
|  _  | |_| | |_) |   \ V / (_| | | |_| | (_| | |_| | (_) | | | |
|_|_|_|____/|____/     \_/ \__,_|_|\__,_|\__,_|\__|_|\___/|_| |_|
 / ___|__ _| | ___ _   _| | __ _| |_ ___  _ __                   
| |   / _` | |/ __| | | | |/ _` | __/ _ \| '__|                  
| |__| (_| | | (__| |_| | | (_| | || (_) | |                     
 \____\__,_|_|\___|\__,_|_|\__,_|\__\___/|_|                     
        """
        self.console.print(banner, style="bold cyan")
    
    def show_main_menu(self):
        """Display simplified main menu options"""
        menu_panel = Panel(
            """[bold green]1.[/bold green] Calculate HDB Price
[bold green]2.[/bold green] View Results History  
[bold green]3.[/bold green] Exit""",
            title="[bold blue]Main Menu[/bold blue]",
            border_style="blue"
        )
        self.console.print(menu_panel)
    
    def load_and_train_model(self):
        """Load data and train polynomial model"""
        with self.console.status("[bold green]Loading data and training polynomial model..."):
            try:
                # Load the real CSV data
                self.model.load_data('sample_data.csv')
                
                # Clean the data
                if self.model.df is not None:
                    cleaned_data = self.processor.clean_data(self.model.df)
                    self.model.df = cleaned_data
                
                # Train the model
                self.model.train_model()
                
                # Show model summary
                metrics = self.model.get_model_metrics()
                polynomial_info = self.model.get_polynomial_equation_info()
                
                summary_table = Table(title="Model Training Summary")
                summary_table.add_column("Metric", style="cyan")
                summary_table.add_column("Value", style="green")
                
                summary_table.add_row("Polynomial Degree", str(polynomial_info['polynomial_degree']))
                summary_table.add_row("Training Samples", f"{metrics['n_samples']:,}")
                summary_table.add_row("Original Features", str(metrics['n_features']))
                summary_table.add_row("Polynomial Features", f"{polynomial_info['n_polynomial_features']:,}")
                summary_table.add_row("Training R²", f"{metrics['train_r2']:.4f}")
                summary_table.add_row("Testing R²", f"{metrics['test_r2']:.4f}")
                summary_table.add_row("Test MAE", f"SGD ${metrics['test_mae']:,.2f}")
                summary_table.add_row("Test RMSE", f"SGD ${metrics['test_rmse']:,.2f}")
                
                self.console.print(summary_table)
                
                rprint("\n[bold green]✅ Model trained successfully![/bold green]")
                
            except Exception as e:
                rprint(f"[bold red]❌ Error: {str(e)}[/bold red]")
    
    def predict_price(self):
        """Predict HDB price with user inputs"""
        if not self.model.is_trained:
            rprint("[bold red]❌ Model not trained. Please train the model first.[/bold red]")
            return
        
        rprint("[bold blue]🏠 Calculate HDB Price[/bold blue]")
        
        try:
            # Get user inputs
            inputs = self.collect_user_inputs()
            
            if inputs:
                # Validate inputs
                is_valid, errors = self.processor.validate_input_data(inputs)
                
                if not is_valid:
                    for error in errors:
                        rprint(f"[bold red]❌ {error}[/bold red]")
                    return
                
                # Make prediction
                prediction, contributions = self.model.predict_price(inputs)
                
                # Display results
                self.display_prediction_results(inputs, prediction, contributions)
                
                # Save to session
                self.session_predictions.append({
                    'inputs': inputs.copy(),
                    'prediction': prediction,
                    'timestamp': pd.Timestamp.now()
                })
                
        except Exception as e:
            rprint(f"[bold red]❌ Error making prediction: {str(e)}[/bold red]")
    
    def collect_user_inputs(self) -> Optional[Dict]:
        """Collect user inputs for prediction"""
        inputs = {}
        
        try:
            # Town selection
            towns = self.model.get_available_towns()
            if not towns:
                rprint("[bold red]❌ No towns available[/bold red]")
                return None
            
            rprint(f"[bold cyan]Towns ({len(towns)} available):[/bold cyan]")
            for i, town in enumerate(towns, 1):  # Show all towns
                rprint(f"  {i}. {town}")
            
            rprint(f"\n[bold yellow]💡 You can type the town number (1-{len(towns)}) or town name[/bold yellow]")
            town_input = Prompt.ask("Select town")
            
            # Check if input is a number (town selection by index)
            try:
                town_choice = int(town_input)
                if 1 <= town_choice <= len(towns):
                    inputs['town'] = towns[town_choice - 1]
                else:
                    rprint(f"[bold red]❌ Please enter a number between 1-{len(towns)}[/bold red]")
                    return None
            except ValueError:
                # Input is a town name - find matching town
                matching_towns = [t for t in towns if town_input.upper() in t.upper()]
                if not matching_towns:
                    rprint("[bold red]❌ Town not found[/bold red]")
                    return None
                elif len(matching_towns) == 1:
                    inputs['town'] = matching_towns[0]
                else:
                    rprint(f"[bold yellow]Multiple matches found:[/bold yellow]")
                    for i, town in enumerate(matching_towns[:5], 1):
                        rprint(f"  {i}. {town}")
                    choice = IntPrompt.ask("Select number", choices=[str(i) for i in range(1, min(6, len(matching_towns) + 1))])
                    inputs['town'] = matching_towns[choice - 1]
            
            # Flat type selection
            flat_types = self.model.get_available_flat_types()
            rprint(f"[bold cyan]Flat Types:[/bold cyan]")
            for i, ft in enumerate(flat_types, 1):
                rprint(f"  {i}. {ft}")
            
            ft_choice = IntPrompt.ask("Select flat type", choices=[str(i) for i in range(1, len(flat_types) + 1)])
            inputs['flat_type'] = flat_types[ft_choice - 1]
            
            # Floor area with validation and guidance
            rprint("[bold cyan]💡 Typical ranges: 2-room (34-64 sqm), 3-room (60-90 sqm), 4-room (80-120 sqm)[/bold cyan]")
            while True:
                area = FloatPrompt.ask("Enter floor area (sqm)")
                if area >= 30 and area <= 300:
                    inputs['floor_area_sqm'] = area
                    break
                else:
                    rprint("[bold red]❌ Floor area must be between 30-300 sqm[/bold red]")
            
            # Storey range selection
            storey_ranges = self.model.get_available_storey_ranges()
            rprint(f"[bold cyan]Storey Ranges:[/bold cyan]")
            for i, sr in enumerate(storey_ranges, 1):
                rprint(f"  {i}. {sr}")
            
            sr_choice = IntPrompt.ask("Select storey range", choices=[str(i) for i in range(1, len(storey_ranges) + 1)])
            inputs['storey_range'] = storey_ranges[sr_choice - 1]
            
            # Flat model selection (reduced to most common models)
            common_models = ['Model A', 'Improved', 'New Generation', 'Premium Apartment', 'Standard', 'Apartment']
            rprint(f"[bold cyan]Flat Models:[/bold cyan]")
            for i, model in enumerate(common_models, 1):
                rprint(f"  {i}. {model}")
            
            model_choice = IntPrompt.ask("Select flat model", choices=[str(i) for i in range(1, len(common_models) + 1)])
            inputs['flat_model'] = common_models[model_choice - 1]
            
            # Remaining lease with validation and guidance
            rprint("[bold cyan]💡 Most HDB flats have 50-90 years remaining lease[/bold cyan]")
            while True:
                lease = IntPrompt.ask("Enter remaining lease (years)")
                if lease >= 45 and lease <= 99:
                    inputs['remaining_lease'] = lease
                    break
                else:
                    rprint("[bold red]❌ Remaining lease must be between 45-99 years for accurate predictions[/bold red]")
            
            return inputs
            
        except KeyboardInterrupt:
            rprint("\n[bold yellow]⚠️ Input cancelled[/bold yellow]")
            return None
    
    def display_prediction_results(self, inputs: Dict, prediction: float, contributions: Dict):
        """Display prediction results with automatic chart generation"""
        # Clear console before showing results
        self.clear_screen()
        self.display_banner()
        
        # Main prediction result
        result_panel = Panel(
            f"[bold green]SGD ${prediction:,.2f}[/bold green]",
            title="[bold blue]Predicted HDB Price[/bold blue]",
            border_style="green"
        )
        self.console.print(result_panel)
        
        # Input summary
        input_table = Table(title="Input Summary")
        input_table.add_column("Property", style="cyan")
        input_table.add_column("Value", style="white")
        
        for key, value in inputs.items():
            display_key = key.replace('_', ' ').title()
            input_table.add_row(display_key, str(value))
        
        self.console.print(input_table)
        
        # Feature contributions (simplified)
        if contributions:
            contrib_table = Table(title="Feature Influence (Scaled)")
            contrib_table.add_column("Feature", style="cyan")
            contrib_table.add_column("Influence", style="green")
            
            # Sort by absolute contribution
            sorted_contrib = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)
            
            for feature, contrib in sorted_contrib[:5]:  # Show top 5
                display_feature = feature.replace('_', ' ').title()
                contrib_table.add_row(display_feature, f"{contrib:+.2f}")
            
            self.console.print(contrib_table)
        
        # Automatically generate personalized visualizations
        rprint("\n[bold cyan]🎨 Generating personalized charts and analysis...[/bold cyan]")
        self.auto_generate_charts(inputs, prediction)
    
    def view_model_performance(self):
        """Display model performance metrics"""
        if not self.model.is_trained:
            rprint("[bold red]❌ Model not trained. Please train the model first.[/bold red]")
            return
        
        metrics = self.model.get_model_metrics()
        polynomial_info = self.model.get_polynomial_equation_info()
        
        # Performance table
        perf_table = Table(title="Polynomial Model Performance")
        perf_table.add_column("Metric", style="cyan")
        perf_table.add_column("Training", style="green")
        perf_table.add_column("Testing", style="yellow")
        
        perf_table.add_row("R² Score", f"{metrics['train_r2']:.4f}", f"{metrics['test_r2']:.4f}")
        perf_table.add_row("MAE (SGD)", f"{metrics['train_mae']:,.2f}", f"{metrics['test_mae']:,.2f}")
        perf_table.add_row("RMSE (SGD)", f"{metrics['train_rmse']:,.2f}", f"{metrics['test_rmse']:,.2f}")
        
        self.console.print(perf_table)
        
        # Model info
        info_text = f"""
Polynomial Degree: {polynomial_info['polynomial_degree']}
Original Features: {metrics['n_features']}
Polynomial Features: {polynomial_info['n_polynomial_features']:,}
Training Samples: {metrics['n_samples']:,}
        """
        
        info_panel = Panel(info_text.strip(), title="Model Information", border_style="blue")
        self.console.print(info_panel)
    
    def generate_visualizations(self):
        """Generate and export visualizations"""
        if not self.model.is_trained:
            rprint("[bold red]❌ Model not trained. Please train the model first.[/bold red]")
            return
        
        with self.console.status("[bold green]Generating visualizations..."):
            try:
                metrics = self.model.get_model_metrics()
                if self.model.df is not None:
                    self.visualizer.create_comprehensive_dashboard(self.model.df, metrics)
                rprint("[bold green]✅ Visualizations created successfully![/bold green]")
                rprint(f"[bold blue]📁 Check the 'graphs/' folder for all visualizations[/bold blue]")
                
            except Exception as e:
                rprint(f"[bold red]❌ Error creating visualizations: {str(e)}[/bold red]")
    
    def view_session_history(self):
        """View session prediction history"""
        if not self.session_predictions:
            rprint("[bold yellow]📝 No predictions made in this session[/bold yellow]")
            return
        
        history_table = Table(title=f"Session History ({len(self.session_predictions)} predictions)")
        history_table.add_column("Time", style="cyan")
        history_table.add_column("Town", style="white")
        history_table.add_column("Type", style="white")
        history_table.add_column("Area (sqm)", style="white")
        history_table.add_column("Predicted Price", style="green")
        
        for pred in self.session_predictions[-10:]:  # Show last 10
            history_table.add_row(
                pred['timestamp'].strftime("%H:%M:%S"),
                pred['inputs']['town'][:20],  # Truncate long names
                pred['inputs']['flat_type'],
                f"{pred['inputs']['floor_area_sqm']:.0f}",
                f"${pred['prediction']:,.0f}"
            )
        
        self.console.print(history_table)
    
    def export_results(self):
        """Export session results and model"""
        if not self.model.is_trained:
            rprint("[bold red]❌ Model not trained. Please train the model first.[/bold red]")
            return
        
        with self.console.status("[bold green]Exporting results..."):
            try:
                # Export model
                self.model.export_model()
                
                # Export data summary
                if self.model.df is not None:
                    self.visualizer.export_data_summary(self.model.df)
                
                # Export session predictions if any
                if self.session_predictions:
                    predictions_df = pd.DataFrame([
                        {
                            'timestamp': pred['timestamp'],
                            'prediction': pred['prediction'],
                            **pred['inputs']
                        }
                        for pred in self.session_predictions
                    ])
                    predictions_df.to_csv('exports/session_predictions.csv', index=False)
                    rprint("📁 Exported session predictions to exports/session_predictions.csv")
                
                rprint("[bold green]✅ Results exported successfully![/bold green]")
                rprint("[bold blue]📁 Check the 'exports/' folder for all files[/bold blue]")
                
            except Exception as e:
                rprint(f"[bold red]❌ Error exporting results: {str(e)}[/bold red]")
    
    def auto_generate_charts(self, inputs: Dict, prediction: float):
        """Automatically generate relevant charts based on user's input"""
        try:
            rprint("📊 Creating price analysis charts...")
            
            # Generate charts specific to the user's selection
            charts_generated = []
            
            # 1. Price distribution for the selected town
            if 'town' in inputs:
                with self.console.status(f"[bold green]Creating price analysis for {inputs['town']}..."):
                    town_chart = self.visualizer.create_town_price_distribution(self.model.df, inputs['town'])
                    if town_chart:
                        charts_generated.append(f"Price distribution for {inputs['town']}")
            
            # 2. Flat type comparison chart
            if 'flat_type' in inputs:
                with self.console.status(f"[bold green]Creating {inputs['flat_type']} analysis..."):
                    flat_type_chart = self.visualizer.create_flat_type_analysis(self.model.df, inputs['flat_type'])
                    if flat_type_chart:
                        charts_generated.append(f"{inputs['flat_type']} market analysis")
            
            # 3. Price vs floor area scatter for similar properties  
            with self.console.status("[bold green]Creating floor area analysis..."):
                area_chart = self.visualizer.create_price_vs_area_chart(self.model.df, inputs)
                if area_chart:
                    charts_generated.append("Price vs floor area analysis")
            
            # 4. Market comparison heatmap
            with self.console.status("[bold green]Creating market heatmap..."):
                heatmap = self.visualizer.create_market_heatmap(self.model.df, inputs['town'], inputs['flat_type'])
                if heatmap:
                    charts_generated.append("Market comparison heatmap")
            
            # Show what was generated
            if charts_generated:
                rprint(f"[bold green]✅ Generated {len(charts_generated)} personalized charts:[/bold green]")
                for chart in charts_generated:
                    rprint(f"  📈 {chart}")
                rprint("[bold blue]📁 Check the 'graphs/' folder to view your charts[/bold blue]")
            else:
                rprint("[bold yellow]⚠️ No charts could be generated with current data[/bold yellow]")
                
        except Exception as e:
            rprint(f"[bold red]❌ Error generating charts: {str(e)}[/bold red]")
    
    def run(self):
        """Main application loop"""
        self.display_banner()
        
        while True:
            try:
                rprint("")  # Empty line for spacing
                self.show_main_menu()
                
                choice = Prompt.ask("\n[bold cyan]Select option[/bold cyan]", 
                                  choices=["1", "2", "3"])
                
                if choice == "1":
                    # Auto-setup on first prediction
                    if not self.model.is_trained:
                        rprint("[bold yellow]Setting up calculator...[/bold yellow]")
                        try:
                            self.model.load_data('sample_data.csv')
                            if self.model.df is not None:
                                cleaned_data = self.processor.clean_data(self.model.df)
                                self.model.df = cleaned_data
                            self.model.train_model()
                            rprint("[bold green]✅ Ready![/bold green]\n")
                        except Exception as e:
                            rprint(f"[bold red]❌ Setup error: {str(e)}[/bold red]")
                            continue
                    self.predict_price()
                elif choice == "2":
                    self.view_session_history()
                elif choice == "3":
                    rprint("\n[bold green]Thank you for using HDB Valuation Calculator![/bold green]")
                    break
                
                # Wait for user to continue
                if choice != "3":
                    input("\nPress Enter to continue...")
                    self.clear_screen()
                    self.display_banner()
                
            except KeyboardInterrupt:
                rprint("\n[bold yellow]Application terminated by user[/bold yellow]")
                break
            except Exception as e:
                rprint(f"[bold red]❌ Unexpected error: {str(e)}[/bold red]")
                input("Press Enter to continue...")
