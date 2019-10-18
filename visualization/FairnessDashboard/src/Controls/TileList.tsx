import React from "react";
import { List } from "office-ui-fabric-react/lib/List";
import { mergeStyleSets } from 'office-ui-fabric-react/lib/Styling';
import { Icon } from "office-ui-fabric-react/lib/Icon";

export interface ITileProp {
    title: string;
    selected: boolean;
    description?: string;
    tags?: string[];
    disabled?: boolean;
    onSelect: () => void;
}

export interface ITileListProps {
    items: ITileProp[];
    columnCount?: number;
}

export class TileList extends React.PureComponent<ITileListProps> {
    private static readonly classNames = mergeStyleSets({
        itemCell: [
          {
            padding: 10,
            height: "250px",
            position: "relative",
            float: "left",
            cursor: "pointer",
            boxSizing: "border-box",
            border: `1px solid grey`,
            selectors: {
              '&:hover': { background: "lightgray" }
            }
          }
        ],
        iconClass: {
            fontSize: "20px",
            position: "absolute",
            right: "10px",
            top: "10px"
        }
    });
    render(): React.ReactNode {
        return (
            <List
                items={this.props.items}
                onRenderCell={this._onRenderCell}
            />
        );
    }

    private _onRenderCell = (item: ITileProp, index: number | undefined): JSX.Element => {
        const columnCount = this.props.columnCount || 3;
        return (
          <div
            className={TileList.classNames.itemCell}
            onClick={item.onSelect.bind(this)}
            data-is-focusable={true}
            style={{
                width: 100 / columnCount + "%"
            }}
          >
            {item.selected && (<Icon iconName="CompletedSolid" className={TileList.classNames.iconClass}/>)}
            <h2>{item.title}</h2>
            <p>{item.description}</p>
          </div>
        );
    }
}