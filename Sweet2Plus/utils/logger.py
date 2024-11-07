from rich.console import Console
from rich.table import Table
from rich.live import Live
import time

console = Console()

class Logger:
    def __init__(self, data):
        self.data = data
        self.step_column = 4
        self.progress_column = 5
        self.live = None  # Placeholder for the Live object

    def generate_table(self):
        """Generate and return a table"""
        table = Table(title="Sweet 2 Plus Status:")

        table.add_column("Cage", justify="left", style="bold cyan", no_wrap=True)
        table.add_column("Subject", justify="left", style="bold green", no_wrap=True)
        table.add_column("Group", justify="left", style="italic magenta ", no_wrap=True)
        table.add_column("Day", justify="left", style="italic yellow", no_wrap=True)
        table.add_column("Step", justify="center", style="bold red", no_wrap=True)
        table.add_column("Progress", justify="right", style="bright_magenta", no_wrap=True)

        if self.data:
            for cage, subject, group, day, step, progress in self.data:
                table.add_row(cage, subject, group, day, step, f"{progress}%")

        return table

    def start_live(self):
        """Start the live display"""
        self.live = Live(self.generate_table(), console=console, refresh_per_second=4)
        self.live.start()

    def stop_live(self):
        """Stop the live display"""
        if self.live:
            self.live.stop()

    def update_table(self, cage, subject, group, day, step, progress):
        """Update the table for Sweet 2 Plus and refresh"""
        # Get matches based on cage, subject, group and date
        matching_indices = [
            index for index, row in enumerate(self.data)
            if row[0] == cage and row[1] == subject and row[2] == group and row[3] == day
        ]

        matching_index = matching_indices[0] if matching_indices else None

        if matching_index is not None:
            # Update that row with new information
            self.data[matching_index][self.step_column] = step
            self.data[matching_index][self.progress_column] = progress

            # Refresh the table with the updated data
            if self.live:
                self.live.update(self.generate_table())
        
        else:
            

if __name__ == '__main__':
    data = [
        ["122", "233", "Group1", "2024-10-15", "Active", "1"],
        ["123", "234", "Group1", "2024-10-14", "Inactive", "2"],
        ["124", "235", "Group2", "2024-10-13", "Active", "3"],
        ["122", "233", "Group1", "2024-10-14", "Active", "4"],
        ["125", "236", "Group2", "2024-10-12", "Inactive", "5"],
        ["126", "237", "Group2", "2024-10-11", "Active", "6"],
        ["122", "238", "Group3", "2024-10-10", "Active", "7"],
        ["127", "239", "Group3", "2024-10-09", "Inactive", "8"],
        ["128", "240", "Group4", "2024-10-08", "Active", "9"],
        ["123", "234", "Group1", "2024-10-07", "Inactive", "10"],
        ["122", "233", "Group1", "2024-10-06", "Active", "11"],
        ["125", "236", "Group2", "2024-10-05", "Inactive", "12"],
        ["126", "237", "Group2", "2024-10-04", "Active", "13"],
        ["124", "235", "Group2", "2024-10-03", "Inactive", "14"],
        ["122", "233", "Group1", "2024-10-02", "Active", "15"]
    ]

    cli_log = Logger(data)
    cli_log.start_live()

    try:
        # Simulate some updates
        time.sleep(2)
        cli_log.update_table("122", "233", "Group1", "2024-10-14", "Dave", "100")
        time.sleep(2)
        cli_log.update_table("124", "235", "Group2", "2024-10-13", "Processing", "75")
        time.sleep(2)

        for i in range(100):
            time.sleep(0.1)
            cli_log.update_table("122", "233", "Group1", "2024-10-02", "TheDaveStep", str(i))
            cli_log.update_table("123", "234", "Group1", "2024-10-02", "TheDaveStep", str(i))
            cli_log.update_table("122", "233", "Group1", "2024-10-02", "TheDaveStep", str(i))
            cli_log.update_table("122", "233", "Group1", "2024-10-02", "TheDaveStep", str(i))

    finally:
        # Ensure the live display stops properly
        cli_log.stop_live()
