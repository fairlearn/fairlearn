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
        container: {
            display: "inline-flex",
            flexDirection: "row",
            flexWrap: "wrap",
            justifyContent: "space-between",
        },
        itemCell: {
            padding: "15px",
            width: "48%",
            position: "relative",
            float: "left",
            cursor: "pointer",
            boxSizing: "border-box",
            backgroundColor: "#FFFFFF",
            marginBottom: "20px",
            marginRight: "0px",
            border: "1px solid #CCCCCC",
            borderRadius: "5px",
            selectors: {
              '&:hover': { background: "lightgray" }
            }
        },
        iconClass: {
            fontSize: "20px",
            position: "absolute",
            left: "15px",
            top: "18px"
        },
        title: {
            color: "#333333",
            fontSize: "18px",
            lineHeight: "22px",
            fontWeight: "normal",
            paddingLeft: "30px",
            margin: 0
        },
        description: {
            paddingTop: "10px",
            paddingLeft: "30px",
            paddingRight: "10px",
            color: "#333333",
            fontSize: "12px",
            lineHeight: "15px",
            fontWeight: "normal",
            textAlign: "justify"
        }
    });
    render(): React.ReactNode {
        return (
            <div className={TileList.classNames.container}>
                {this.props.items.map((item, index) => this._onRenderCell(item, index))}
            </div>
        );
    }

    private _onRenderCell = (item: ITileProp, index: number | undefined): JSX.Element => {
        const columnCount = this.props.columnCount || 3;
        return (
          <div
            className={TileList.classNames.itemCell}
            onClick={item.onSelect.bind(this)}
            key={index}
            data-is-focusable={true}>
            <Icon iconName={item.selected ? "RadioBtnOn" : "RadioBtnOff"} className={TileList.classNames.iconClass}/>
            <h2 className={TileList.classNames.title}>{item.title}</h2>
            <p className={TileList.classNames.description}>{item.description}</p>
          </div>
        );
    }
}