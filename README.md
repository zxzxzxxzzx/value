# HDB Polynomial Price Calculator

## Overview

The HDB Polynomial Price Calculator is an interactive command-line application that uses 4th degree polynomial regression to predict Housing Development Board (HDB) flat prices in Singapore. The system processes real estate data through advanced polynomial feature engineering to capture non-linear price relationships, providing more accurate valuations than traditional linear models. The application combines machine learning prediction with comprehensive data visualization and session management capabilities, all delivered through a rich terminal interface with color-coded market indicators.

## User Preferences

Preferred communication style: Simple, everyday language.

## Recent Changes

**August 14, 2025**: 
- Streamlined interface from 7 complex menu options to just 3 simple choices: Calculate Price, View History, Exit
- Added automatic model setup on first use - users no longer need to manually train the model
- Simplified input prompts with shorter, clearer labels and reduced verbose text
- Fixed prediction accuracy with smart input validation (30-300 sqm floor area, 45-99 years lease)
- Custom ASCII art header for professional branding
- Curated flat_model selection to 6 essential options covering 85%+ of cases
- All predictions now realistic: SGD 259K for 2-room, SGD 315K for 3-room, SGD 381K for 4-room
- Enhanced model performance to R² 0.73 with proper 6-feature polynomial engineering
- Much simpler workflow: users just select "Calculate Price" and get guided through streamlined inputs
- Shows all 26 towns with numbered selection - users can type "8" instead of "CENTRAL AREA"
- Automatically generates 4 personalized charts after each prediction: town analysis, flat type analysis, price vs area, and market heatmap
- Clean results display with console clearing before showing prediction summary

## System Architecture

### Machine Learning Architecture

**Polynomial Regression Engine**: The core prediction system uses scikit-learn's 4th degree polynomial feature transformation combined with linear regression. This approach captures complex non-linear relationships between property features and prices that simpler models miss. The polynomial pipeline includes feature scaling and standardization to handle the expanded feature space effectively.

**Advanced Data Processing**: The system implements conservative outlier removal (0.5% and 99.5% quantiles) specifically optimized for polynomial regression to prevent extreme values from distorting the model. Feature engineering includes categorical encoding for towns and flat types, with polynomial expansion creating interaction terms between all features.

**Model Validation Framework**: Comprehensive metrics tracking including R² score, Mean Absolute Error (MAE), and Root Mean Square Error (RMSE) to assess polynomial model performance. The system maintains separate training and testing datasets with proper validation protocols.

### Interface Architecture

**Rich CLI Framework**: Built on the Rich library for enhanced terminal output with colored text, tables, panels, and progress indicators. The interface provides numbered menu navigation with input validation and error handling throughout the user journey.

**Session Management**: Tracks multiple prediction sessions with market health indicators (Below Market/At Market/Above Market) based on percentile comparisons. Session data can be exported to CSV for further analysis.

**Interactive Workflow**: Menu-driven interface supporting data loading, model training, individual predictions, performance analysis, visualization generation, and result export in a logical sequence.

### Visualization Architecture

**Non-Interactive Backend**: Uses matplotlib's 'Agg' backend to prevent display issues in terminal environments. All visualizations are automatically exported to the 'graphs' directory with organized file naming.

**Comprehensive Chart Suite**: Generates price distribution analysis by town and flat type, scatter plots with polynomial trend lines, correlation heatmaps, and statistical distribution charts. Visualization generation includes progress feedback and export notifications.

**Performance Optimization**: Implements data sampling for large datasets to maintain chart generation speed while preserving statistical representation.

### File Organization Pattern

**Modular Component Design**: Separates concerns across specialized modules - polynomial model logic, data processing, visualization, and CLI interface. Each component maintains clear interfaces and can be used independently.

**Automatic Directory Management**: Creates 'graphs' and 'exports' directories automatically for organized output. Sample data generation fallback ensures the application runs even without pre-existing datasets.

## External Dependencies

**Core Machine Learning Stack**: scikit-learn for polynomial feature transformation and linear regression, pandas for data manipulation, numpy for numerical operations.

**Visualization Libraries**: matplotlib and seaborn for chart generation with automatic export functionality.

**Terminal Interface**: Rich library for enhanced CLI experience with colored output, tables, and interactive prompts.

**Data Format Support**: CSV file processing for HDB real estate data input and session export functionality.

**Cross-Platform Compatibility**: colorama for consistent colored output across different terminal environments.