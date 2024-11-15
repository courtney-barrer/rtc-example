from textual.app import App
from textual.widgets import DataTable, RichLog, Footer, Header
from textual.reactive import reactive
import baldr as ba

class ComponentInfoTUI(App):
    """A Textual TUI app to display and refresh ComponentInfo details."""

    REFRESH_INTERVAL = 1  # Refresh interval in seconds

    # Reactive data source
    components_data = reactive({})

    cim = ba.ComponentInfoManager.get()
    command_choices = ["run", "step", "pause", "exit"]

    def __init__(self):
        super().__init__()
        self.data_table = DataTable()
        self.logging_widget = RichLog()

    def compose(self):
        """Set up the layout and widgets."""
        # Initialize DataTable to show component data
        self.data_table.add_column("Name", key="name")
        self.data_table.add_column("Command", key="cmd")
        self.data_table.add_column("Status", key="status")
        self.data_table.add_column("Loop Count", key="loop_count")
        self.data_table.add_column("PID", key="pid")

        # Add DataTable to the application layout
        yield Header()
        yield Footer()
        yield self.data_table
        yield self.logging_widget

    async def on_mount(self) -> None:
        """Initialize periodic refresh."""
        # await self.view.dock(self.data_table, edge="top")
        # await self.view.dock(self.logging_widget, edge="bottom")
        # Start the periodic refresh
        self.set_interval(self.REFRESH_INTERVAL, self.refresh_data)

    def local_log(self, message: str):
        self.logging_widget.write(message)

    async def refresh_data(self) -> None:
        """Fetch and refresh the component data."""
        # Get the ComponentInfoManager and components
        components = self.cim.components

        # Update existing rows and add new rows
        for name, component in self.cim.components.items():
            # Check if the key is in the DataTable
            if name in self.data_table.rows:
                self.data_table.update_cell(name, "cmd"       , component.cmd.name)
                self.data_table.update_cell(name, "status"    , component.status.name)
                self.data_table.update_cell(name, "loop_count", str(component.loop_count))
                self.data_table.update_cell(name, "pid"       , str(component.pid))
            else:
                self.local_log(f"Adding new row {name}")

                self.data_table.add_row(
                    name,
                    component.cmd.name,
                    component.status.name,
                    str(component.loop_count),
                    str(component.pid),
                    key=name
                )


def main():
    ComponentInfoTUI().run()

if __name__ == "__main__":
    main()
