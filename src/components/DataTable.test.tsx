import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { DataTable, Column } from "./DataTable";

// Mock translations
vi.mock("@/contexts/LanguageContext", () => ({
  useUITranslation: () => ({
    tUI: (key: string) => {
      const translations: Record<string, string> = {
        "common.showing": "Showing",
        "common.of": "of",
        "common.noData": "No data available",
      };
      return translations[key] || key;
    },
  }),
}));

interface TestItem {
  id: string;
  name: string;
  age: number;
  createdAt: Date;
}

const testData: TestItem[] = [
  { id: "1", name: "Alice", age: 30, createdAt: new Date("2024-01-01") },
  { id: "2", name: "Bob", age: 25, createdAt: new Date("2024-02-01") },
  { id: "3", name: "Charlie", age: 35, createdAt: new Date("2024-03-01") },
  { id: "4", name: "David", age: 28, createdAt: new Date("2024-04-01") },
  { id: "5", name: "Eve", age: 32, createdAt: new Date("2024-05-01") },
];

const columns: Column<TestItem>[] = [
  {
    key: "name",
    header: "Name",
    sortable: true,
    sortValue: (item) => item.name,
    render: (item) => <span>{item.name}</span>,
  },
  {
    key: "age",
    header: "Age",
    sortable: true,
    sortValue: (item) => item.age,
    render: (item) => <span>{item.age}</span>,
  },
  {
    key: "created",
    header: "Created",
    sortable: true,
    sortValue: (item) => item.createdAt.getTime(),
    render: (item) => <span>{item.createdAt.toLocaleDateString()}</span>,
  },
];

describe("DataTable", () => {
  describe("rendering", () => {
    it("renders table headers correctly", () => {
      render(
        <DataTable
          data={testData}
          columns={columns}
          keyExtractor={(item) => item.id}
        />,
      );

      expect(screen.getByText("Name")).toBeInTheDocument();
      expect(screen.getByText("Age")).toBeInTheDocument();
      expect(screen.getByText("Created")).toBeInTheDocument();
    });

    it("renders data rows correctly", () => {
      render(
        <DataTable
          data={testData}
          columns={columns}
          keyExtractor={(item) => item.id}
          pageSize={10}
        />,
      );

      expect(screen.getByText("Alice")).toBeInTheDocument();
      expect(screen.getByText("Bob")).toBeInTheDocument();
      expect(screen.getByText("30")).toBeInTheDocument();
      expect(screen.getByText("25")).toBeInTheDocument();
    });

    it("shows empty message when no data", () => {
      render(
        <DataTable
          data={[]}
          columns={columns}
          keyExtractor={(item) => item.id}
          emptyMessage="No items found"
        />,
      );

      expect(screen.getByText("No items found")).toBeInTheDocument();
    });

    it("shows default empty message when no custom message provided", () => {
      render(
        <DataTable
          data={[]}
          columns={columns}
          keyExtractor={(item) => item.id}
        />,
      );

      expect(screen.getByText("No data available")).toBeInTheDocument();
    });
  });

  describe("pagination", () => {
    it("paginates data correctly with custom page size", () => {
      render(
        <DataTable
          data={testData}
          columns={columns}
          keyExtractor={(item) => item.id}
          pageSize={2}
        />,
      );

      // Should show first 2 items
      expect(screen.getByText("Alice")).toBeInTheDocument();
      expect(screen.getByText("Bob")).toBeInTheDocument();
      expect(screen.queryByText("Charlie")).not.toBeInTheDocument();

      // Should show pagination info
      expect(screen.getByText(/Showing 1-2/)).toBeInTheDocument();
      expect(screen.getByText(/of 5/)).toBeInTheDocument();
    });

    it("navigates to next page", () => {
      render(
        <DataTable
          data={testData}
          columns={columns}
          keyExtractor={(item) => item.id}
          pageSize={2}
        />,
      );

      const nextButton = screen.getByLabelText("Next page");
      fireEvent.click(nextButton);

      expect(screen.queryByText("Alice")).not.toBeInTheDocument();
      expect(screen.getByText("Charlie")).toBeInTheDocument();
      expect(screen.getByText("David")).toBeInTheDocument();
    });

    it("navigates to previous page", () => {
      render(
        <DataTable
          data={testData}
          columns={columns}
          keyExtractor={(item) => item.id}
          pageSize={2}
        />,
      );

      // Go to page 2
      const nextButton = screen.getByLabelText("Next page");
      fireEvent.click(nextButton);

      // Go back to page 1
      const prevButton = screen.getByLabelText("Previous page");
      fireEvent.click(prevButton);

      expect(screen.getByText("Alice")).toBeInTheDocument();
      expect(screen.getByText("Bob")).toBeInTheDocument();
    });

    it("navigates to first page", () => {
      render(
        <DataTable
          data={testData}
          columns={columns}
          keyExtractor={(item) => item.id}
          pageSize={2}
        />,
      );

      // Go to page 3
      const nextButton = screen.getByLabelText("Next page");
      fireEvent.click(nextButton);
      fireEvent.click(nextButton);

      // Go to first page
      const firstButton = screen.getByLabelText("First page");
      fireEvent.click(firstButton);

      expect(screen.getByText("Alice")).toBeInTheDocument();
    });

    it("navigates to last page", () => {
      render(
        <DataTable
          data={testData}
          columns={columns}
          keyExtractor={(item) => item.id}
          pageSize={2}
        />,
      );

      const lastButton = screen.getByLabelText("Last page");
      fireEvent.click(lastButton);

      expect(screen.getByText("Eve")).toBeInTheDocument();
      expect(screen.queryByText("Alice")).not.toBeInTheDocument();
    });

    it("disables previous button on first page", () => {
      render(
        <DataTable
          data={testData}
          columns={columns}
          keyExtractor={(item) => item.id}
          pageSize={2}
        />,
      );

      const prevButton = screen.getByLabelText("Previous page");
      const firstButton = screen.getByLabelText("First page");

      expect(prevButton).toBeDisabled();
      expect(firstButton).toBeDisabled();
    });

    it("disables next button on last page", () => {
      render(
        <DataTable
          data={testData}
          columns={columns}
          keyExtractor={(item) => item.id}
          pageSize={2}
        />,
      );

      const lastButton = screen.getByLabelText("Last page");
      fireEvent.click(lastButton);

      const nextButton = screen.getByLabelText("Next page");
      expect(nextButton).toBeDisabled();
      expect(lastButton).toBeDisabled();
    });

    it("does not show pagination when all items fit on one page", () => {
      render(
        <DataTable
          data={testData}
          columns={columns}
          keyExtractor={(item) => item.id}
          pageSize={10}
        />,
      );

      expect(screen.queryByLabelText("Next page")).not.toBeInTheDocument();
    });

    it("resets to page 1 when data changes", () => {
      const { rerender } = render(
        <DataTable
          data={testData}
          columns={columns}
          keyExtractor={(item) => item.id}
          pageSize={2}
        />,
      );

      // Go to page 2
      const nextButton = screen.getByLabelText("Next page");
      fireEvent.click(nextButton);
      expect(screen.getByText("Charlie")).toBeInTheDocument();

      // Change data
      const newData = testData.slice(0, 3);
      rerender(
        <DataTable
          data={newData}
          columns={columns}
          keyExtractor={(item) => item.id}
          pageSize={2}
        />,
      );

      // Should be back on page 1
      expect(screen.getByText("Alice")).toBeInTheDocument();
    });
  });

  describe("sorting", () => {
    it("sorts by column ascending on first click", () => {
      render(
        <DataTable
          data={testData}
          columns={columns}
          keyExtractor={(item) => item.id}
          pageSize={10}
        />,
      );

      const ageHeader = screen.getByText("Age");
      fireEvent.click(ageHeader);

      // Get all age cells
      const rows = screen.getAllByRole("row").slice(1); // Skip header row
      const ages = rows.map((row) => row.textContent);

      // Bob (25) should be first
      expect(ages[0]).toContain("Bob");
      expect(ages[0]).toContain("25");
    });

    it("sorts by column descending on second click", () => {
      render(
        <DataTable
          data={testData}
          columns={columns}
          keyExtractor={(item) => item.id}
          pageSize={10}
        />,
      );

      const ageHeader = screen.getByText("Age");
      fireEvent.click(ageHeader); // First click - ascending
      fireEvent.click(ageHeader); // Second click - descending

      const rows = screen.getAllByRole("row").slice(1);
      const ages = rows.map((row) => row.textContent);

      // Charlie (35) should be first
      expect(ages[0]).toContain("Charlie");
      expect(ages[0]).toContain("35");
    });

    it("removes sort on third click", () => {
      render(
        <DataTable
          data={testData}
          columns={columns}
          keyExtractor={(item) => item.id}
          pageSize={10}
        />,
      );

      const ageHeader = screen.getByText("Age");
      fireEvent.click(ageHeader); // First click - ascending
      fireEvent.click(ageHeader); // Second click - descending
      fireEvent.click(ageHeader); // Third click - no sort

      const rows = screen.getAllByRole("row").slice(1);

      // Should be back to original order (Alice first)
      expect(rows[0].textContent).toContain("Alice");
    });

    it("switches sort column when clicking different header", () => {
      render(
        <DataTable
          data={testData}
          columns={columns}
          keyExtractor={(item) => item.id}
          pageSize={10}
        />,
      );

      const ageHeader = screen.getByText("Age");
      const nameHeader = screen.getByText("Name");

      fireEvent.click(ageHeader); // Sort by age ascending
      fireEvent.click(nameHeader); // Sort by name ascending

      const rows = screen.getAllByRole("row").slice(1);

      // Alice should be first when sorted by name
      expect(rows[0].textContent).toContain("Alice");
    });

    it("does not sort non-sortable columns", () => {
      const nonSortableColumns: Column<TestItem>[] = [
        {
          key: "name",
          header: "Name",
          sortable: false,
          render: (item) => <span>{item.name}</span>,
        },
      ];

      render(
        <DataTable
          data={testData}
          columns={nonSortableColumns}
          keyExtractor={(item) => item.id}
          pageSize={10}
        />,
      );

      const nameHeader = screen.getByText("Name");
      const initialRows = screen.getAllByRole("row").slice(1);
      const initialOrder = initialRows.map((row) => row.textContent);

      fireEvent.click(nameHeader);

      const afterClickRows = screen.getAllByRole("row").slice(1);
      const afterClickOrder = afterClickRows.map((row) => row.textContent);

      expect(initialOrder).toEqual(afterClickOrder);
    });
  });

  describe("row styling", () => {
    it("applies custom row class name", () => {
      render(
        <DataTable
          data={testData}
          columns={columns}
          keyExtractor={(item) => item.id}
          pageSize={10}
          rowClassName={(item) =>
            item.age > 30 ? "highlight-row" : "normal-row"
          }
        />,
      );

      const rows = screen.getAllByRole("row").slice(1);

      // Alice (30) should not have highlight
      expect(rows[0]).toHaveClass("normal-row");

      // Charlie (35) should have highlight
      expect(rows[2]).toHaveClass("highlight-row");
    });
  });
});
