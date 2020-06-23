import { getTheme, IProcessedStyleSet, IStyle, mergeStyleSets } from 'office-ui-fabric-react';

export interface IDataSpecificationBladeStyles {
    title: IStyle;
    frame: IStyle;
    text: IStyle;
}

export const DataSpecificationBladeStyles: () => IProcessedStyleSet<IDataSpecificationBladeStyles> = () => {
    const theme = getTheme();
    return mergeStyleSets<IDataSpecificationBladeStyles>({
        title: {
            color: theme.semanticColors.bodyText,
            height: '20px',
            paddingBottom: '10px',
        },
        frame: {
            paddingTop: '35px',
            paddingLeft: '60px',
            width: '120px',
            boxSizing: 'content-box',
        },
        text: {
            color: theme.semanticColors.bodyText,
        },
    });
};
